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

import concurrent.futures as cf
from collections import deque
import time
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import psutil
import yaml
from tokenizers import Tokenizer

from . import layer_runner, telemetry
from .config import validate_expert_top_k_by_layer
from .kv_cache import KVCache, SteppedKVCache, fork_hybrid_kv_endpoint
from .model_loader import WeightStore
from .prefetcher import Prefetcher
from .sampler import SamplingParams, sample
from .weight_cache import WeightCache


_HYBRID_RECURRENT_MODEL_TYPES = frozenset({
    "kimi_linear", "kimi_k3", "qwen3_5_moe", "qwen3_5",
    "qwen4_exp", "jet_nemotron", "lfm2", "glm5_next",
})

# Deliberately sparse: these are the already-supported Qwen hybrid prefill
# ladder rungs, plus 0 for the live automatic policy.  Keeping the accepted
# set exact prevents a profile typo from silently selecting a never-gated
# Metal shape on the 16 GB host.
QWEN35_PREFILL_CHUNK_CEILINGS = frozenset((0, 1, 8, 32, 128, 512))
# Two real, unchanged request shapes set this content-blind phase boundary:
# 16,029 prompt tokens improved 5.65%, while 6,339 tokens regressed 1.0%.
# Keep it implementation-stable and explicit until a wider corpus justifies
# either a tunable policy or a default-on profile.
QWEN35_PHASE_HEAD_MIN_PROMPT_TOKENS = 8192


_MEMORY_RETRY_HARD_CAP_RE = re.compile(
    r"hard Metal cap: phase=(?P<subphase>[a-zA-Z0-9_-]+) "
    r"layer=(?P<layer>\d+) tokens=(?P<tokens>\d+) "
    r"observed=(?P<observed>\d+) limit=(?P<limit>\d+)")


def _lossless_16bit_host_spool(value: mx.array, *, expected_dtype):
    """Copy a released FP16/BF16 activation to CPU without conversion."""
    import numpy as np

    if expected_dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(
            "lossless host spool supports only FP16/BF16 activations, "
            f"got {expected_dtype}")
    if value.dtype != expected_dtype:
        raise TypeError(
            "lossless host spool activation dtype changed: "
            f"expected {expected_dtype}, got {value.dtype}")
    mx.eval(value)
    return np.array(
        np.asarray(value.view(mx.uint16)), dtype=np.uint16, copy=True)


def _restore_lossless_16bit_host_spool(value, *, dtype) -> mx.array:
    """Restore an exact raw FP16/BF16 CPU payload to its released dtype."""
    if dtype not in (mx.float16, mx.bfloat16):
        raise TypeError(
            "lossless host spool supports only FP16/BF16 activations, "
            f"got {dtype}")
    result = mx.array(value, dtype=mx.uint16).view(dtype)
    mx.eval(result)
    return result


def _memory_retry_diagnostic(error: MemoryError) -> dict[str, object]:
    """Return only content-free fields from a known memory refusal.

    Exception text stays inside the engine because it may grow contextual
    details later. The strict pattern admits only the operator phase label
    and integers already produced by the hard-cap guards.
    """
    match = _MEMORY_RETRY_HARD_CAP_RE.search(str(error))
    if match is None:
        return {}
    return {
        "retry_reason": "hard_metal_cap",
        "retry_subphase": match.group("subphase"),
        "retry_layer": int(match.group("layer")),
        "retry_completed_tokens": int(match.group("tokens")),
        "retry_observed_metal_bytes": int(match.group("observed")),
        "retry_metal_limit_bytes": int(match.group("limit")),
    }


def _glm53_expanded_prefill_cache(kv):
    """Create exact dense-prefix K/V; sparse absorbed MLA drops it per layer."""
    return {}


def _glm53_release_expanded_prefill_layer(kv, layer: int) -> None:
    """Release one expanded layer when that optional cache exists."""
    cache = getattr(kv, "_glm53_expanded_prefill", None)
    if cache is not None:
        cache.pop(int(layer), None)


def qwen35_phase_head_request_active(
    enabled: bool,
    prompt_tokens: int,
    min_prompt_tokens: int = QWEN35_PHASE_HEAD_MIN_PROMPT_TOKENS,
) -> bool:
    """Content-blind admission for the measured long-context head lifecycle."""
    return bool(
        enabled
        and int(prompt_tokens) >= int(min_prompt_tokens))


def attach_hybrid_recurrent_cache(
    kv,
    *,
    model_type: str,
    num_hidden_layers: int,
    kda_spill_dir: str = "",
):
    """Attach the fixed-size recurrent companion to any hybrid KV backend.

    ``new_kv()`` is normally the canonical factory, but the exact paged-KV
    admission path constructs :class:`PagedKVCache` directly after it has
    decided the resident cache is unsafe.  Keeping this attachment in one
    helper prevents that alternate backend from silently running every
    DeltaNet/KDA tile with ``state_cache=None`` and therefore resetting the
    released recurrence at each tile boundary.
    """
    if model_type not in _HYBRID_RECURRENT_MODEL_TYPES:
        return kv
    if getattr(kv, "kda_cache", None) is None:
        from .kda_state import KDAStateCache

        kv.kda_cache = KDAStateCache(int(num_hidden_layers))
    if model_type == "kimi_k3" and kda_spill_dir:
        kv.kda_cache.enable_disk_spill(kda_spill_dir)
    return kv


def _fork_matched_hybrid_stable_boundary(
    kv,
    *,
    matched_tokens: int,
    stable_boundary_tokens: int,
    prompt_tokens: int,
):
    """Retain an already-matched recurrent boundary before suffix mutation."""
    matched = int(matched_tokens)
    stable = int(stable_boundary_tokens)
    total = int(prompt_tokens)
    if (type(kv) in (KVCache, SteppedKVCache)
            and matched == stable
            and 0 < stable < total):
        return fork_hybrid_kv_endpoint(kv)
    return None


def _stable_boundary_persistence_allowed(persistence, *, approximate: bool) -> bool:
    """Whether a stable boundary is representable by this disk format."""
    if persistence is None:
        return False
    return bool(
        approximate
        or not getattr(
            persistence, "requires_approximate_stable_prefix", False)
    )


def _prefer_longer_persisted_hybrid_prefix(
    *, model_type: str, best_case: str,
) -> bool:
    """Whether disk may replace a shorter resident recurrent extension.

    Recurrent state cannot be trimmed to an arbitrary longer common prefix.
    These Qwen families persist every auxiliary recurrent component required
    to restore a checksum- and fingerprint-validated exact endpoint, so a
    strictly longer disk match is preferable to replaying the resident
    boundary's suffix.  Ordinary repeats/branches keep the resident path.
    """
    return bool(
        best_case == "extension"
        and model_type in ("qwen3_5", "qwen3_5_moe", "qwen4_exp"))


def hybrid_prefill_chunk_size(available_bytes: int, model_scale: int = 0) -> int:
    """F94/F95: descending chunk-size ladder for qwen3_5/qwen3_5_moe's fixed,
    hot_prompt_kv-compatible prefill chunk (GLM's live adaptive_chunk_size
    resampling is not an option here -- see the callers' comments).

    History: a single low/high binary split (512 vs 128) was proven
    insufficient live (F94, 2026-07-20/21) -- a real lossy-Qwen3.6-27B
    request reproducibly hit "unsafe Metal reservation refused" at
    chunk=128, and the wider descending ladder was STILL insufficient once
    this server started caching and reusing ONE engine across every
    subsequent request forever: a healthy available_bytes reading taken
    once, at a server's first-ever request, has no bearing on real memory
    conditions hours later, so the 512/128 tiers were removed entirely and
    32 became the ceiling for everything, regardless of size or reading.

    F95 (2026-07-21): reinstated the wider tiers (512/128), because the
    actual problem was never "chunk=512 is unsafe" -- it was "deciding it
    ONCE, engine-wide, for a lifetime memory can drift arbitrarily far
    over" that was unsafe. This function is now called PER CONVERSATION
    (see StreamingEngine.generate()'s hot_prompt_kv slot handling): a fresh
    conversation with no matching hot-KV slot samples live memory right
    then and picks a chunk size that only has to stay valid for that ONE
    conversation's own lifetime (typically minutes), not the whole
    server's. A conversation that already has a matching slot reuses
    THAT slot's own recorded chunk size instead of resampling at all,
    preserving the fixed-chunk invariant hot_prompt_kv needs within one
    reuse lineage. `model_scale` (hidden*intermediate) is no longer
    consulted (a direct probe found chunk size barely moves
    _layer_transient for a fixed model size anyway, ~7% from 128 to 32;
    the real lever was always the weight-cache floor, see
    hybrid_min_weight_cache_floor_mb) -- kept only so existing callers
    don't need to change shape."""
    if available_bytes >= 4_000_000_000:
        return 512
    if available_bytes >= 2_000_000_000:
        return 128
    if available_bytes >= 1_000_000_000:
        return 32
    if available_bytes >= 500_000_000:
        return 8
    return 1


def kimi_k3_prompt_prefill_schedule(
    prompt_tokens: int,
    *,
    policy: str,
    long_context_tokens: int,
    short_tile_width: int,
    long_tile_width: int,
    short_dense_mlp_tile_size: int,
    long_dense_mlp_tile_size: int,
) -> tuple[int, int, str]:
    """Select K3's content-blind prefill schedule from rendered token count.

    ``fixed`` preserves the historical explicit tile settings.  The opt-in
    ``prompt-length`` policy uses the short schedule below one configurable
    token boundary and the long schedule at/above it.  It deliberately cannot
    inspect prompt text, messages, tools, routes, or subjects.
    """
    if policy == "fixed":
        return long_tile_width, long_dense_mlp_tile_size, "fixed"
    if policy != "prompt-length":
        raise ValueError(
            "K3 prefill tile policy must be 'fixed' or 'prompt-length'")
    if prompt_tokens < long_context_tokens:
        return short_tile_width, short_dense_mlp_tile_size, "short"
    return long_tile_width, long_dense_mlp_tile_size, "long"


def kimi_k3_prefill_schedule_compatible(
    *, policy: str, active_schedule: str, cached_schedule: str,
) -> bool:
    """Whether an in-memory K3 endpoint can serve this tile schedule."""
    return policy != "prompt-length" or cached_schedule == active_schedule


def hybrid_min_weight_cache_floor_mb(available_bytes: int) -> int:
    """F94: the weight-cache floor for qwen3_5/qwen3_5_moe -- deliberately
    NOT gated on `available_bytes` (the parameter is kept only so existing
    callers/tests didn't need to change shape). Live-reproduced twice: a
    real Qwen3.6-27B request constructed its engine at a HEALTHY
    available=10.27GB (picking chunk=512, and the OLD code's floor=1.5GB
    default), then still hit "unsafe Metal reservation refused" ~3s later
    at available=2.80GB -- a single 30K-token/64-layer dense sweep drained
    7.5GB of system memory over its OWN lifetime (growing weight cache,
    growing KV, quantize-on-load scratch) before the failing layer was
    even reached. A one-time construction-time reading cannot predict
    that, so gating the floor on it is fighting the wrong variable --
    unlike prefill_chunk_size (which hot_prompt_kv genuinely requires
    fixed for the whole sweep, a real constraint each conversation still
    has), the cache floor has no such requirement and can just always be
    conservative, engine-wide, regardless of per-conversation chunk size.
    This costs nothing when memory is plentiful (the cache still grows to
    max_weight_cache_mb under ordinary LRU eviction; the floor only bites
    during an actual reserve() squeeze) and buys the maximum possible
    headroom exactly when a squeeze happens."""
    return 64


def compact_expert_io_batch_size(
        page_bytes: int, cache_bytes: int, *,
        target_bytes: int = 268_435_456, max_batch: int = 16) -> int:
    """Choose a bounded I/O coalescing batch from representation byte size.

    The old fail-closed MoE default fetched one expert at a time. That was
    necessary for ~75 MB BF16 expert pages, but it needlessly turns a native
    compact checkpoint into hundreds of small ``store.fetch`` transactions.
    Pick the smallest integer batch whose payload reaches a useful coalescing
    target, while limiting that target to one eighth of the configured
    weight-cache budget and retaining a hard page-count ceiling.

    With page size ``P`` and byte goal
    ``G = max(P, min(target, cache/8))``, ``B = ceil(G/P)`` gives
    ``G <= B*P < G+P`` before the page-count cap. The existing live governor
    remains authoritative and may reduce ``B`` further at every compute batch;
    this helper only raises the attempted ceiling when the checkpoint already
    stores the expert in a compact native representation.
    """
    page_bytes = int(page_bytes)
    cache_bytes = int(cache_bytes)
    target_bytes = int(target_bytes)
    max_batch = int(max_batch)
    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    if cache_bytes < 0:
        raise ValueError("cache_bytes must be non-negative")
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    if max_batch <= 0:
        raise ValueError("max_batch must be positive")

    byte_goal = max(
        page_bytes,
        min(target_bytes, cache_bytes // 8),
    )
    return min(
        max_batch,
        max(1, (byte_goal + page_bytes - 1) // page_bytes),
    )


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


def _layer_transient_reserve_margin(position_count: int) -> int:
    """Secondary F42 pad for a measured layer-compute high-water mark.

    ``MemoryGovernor.current_ceiling`` already preserves its critical system
    reserve (1.2 GB by default).  Multi-position prefill keeps the additional
    400 MB estimator/allocator pad because a following layer can have a larger
    shape-dependent scratch requirement than the observations seen so far.

    A one-position decode sweep is different: ``_layer_transient`` is the
    maximum observed scratch over the complete prompt sweep and every prior
    decode layer, so adding the same 400 MB again can reject an operation even
    after all reclaimable weights have been evicted and the measured high-water
    itself fits below the hard live ceiling.  Use that measured maximum as the
    fail-closed reservation while retaining the governor's independent critical
    reserve.  This changes admission only; model arithmetic is untouched.
    """
    if position_count <= 0:
        raise ValueError("position_count must be positive")
    return 0 if position_count == 1 else 400_000_000


def _recurring_layer_transient_reserve_margin(
        position_count: int, observation_count: int,
) -> int:
    """Retire the secondary shape-uncertainty pad after recurrence proof.

    The governor's independent critical reserve remains in force.  This only
    removes the extra 400-MB allowance once the same typed layer signature and
    position count have completed once.  The first occurrence necessarily ran
    before a learned reserve existed; after it succeeds, the measured
    high-water is reserved in full and monotonically tracks the recurring
    maximum.  Keeping a pad only on occurrence two is therefore not a
    first-allocation safety guarantee, while the governor's independent 1.2-GB
    critical reserve continues to protect system memory.
    """
    if observation_count < 0:
        raise ValueError("observation_count must be non-negative")
    if observation_count >= 1:
        return 0
    return _layer_transient_reserve_margin(position_count)


def _layer_transient_for_positions(
        position_count: int, prefill_bytes: int, decode_bytes: int,
) -> tuple[int, int]:
    """Return the shape-class high-water and its independent safety pad."""
    margin = _layer_transient_reserve_margin(position_count)
    transient = decode_bytes if position_count == 1 else prefill_bytes
    return max(0, int(transient)), margin


def _remaining_layer_transient_reserve(
        measured_bytes: int, completed_output_bytes: int,
) -> int:
    """Avoid charging already-live dense output twice during a layer.

    The measured layer high-water includes the prompt-sized list of evaluated
    output tiles that exists immediately before their final concatenation.
    During the next occurrence, those completed tiles are already included in
    ``mx.get_active_memory()``. Subtract only their byte-exact payload from the
    historical reserve; every other measured allocation remains reserved.
    """
    measured = int(measured_bytes)
    completed = int(completed_output_bytes)
    if measured < 0 or completed < 0:
        raise ValueError("transient and completed output bytes must be non-negative")
    return max(0, measured - completed)


def _cache_io_snapshot(engine) -> tuple[int, ...]:
    """Cumulative counters used to derive one request's physical work."""
    stats = engine.cache.stats
    governor = getattr(engine, "governor", None)
    stage_snapshot = getattr(engine.store, "stage_snapshot", None)
    store_stages = (
        stage_snapshot() if callable(stage_snapshot) else (0, 0, 0, 0))
    glm53_fp8_snapshot = getattr(engine.store, "glm53_fp8_snapshot", None)
    glm53_fp8_stages = (
        glm53_fp8_snapshot()
        if callable(glm53_fp8_snapshot) else (0, 0, 0, 0, 0, 0, 0, 0)
    )
    glm53_fp8_direct_snapshot = getattr(
        engine.store, "glm53_fp8_direct_snapshot", None)
    glm53_fp8_direct_stages = (
        glm53_fp8_direct_snapshot()
        if callable(glm53_fp8_direct_snapshot)
        else (0, 0, 0, 0, 0, 0, 0, 0)
    )
    qwen4_fp8_direct_snapshot = getattr(
        engine.store, "qwen4_fp8_direct_snapshot", None)
    qwen4_fp8_direct_stages = (
        qwen4_fp8_direct_snapshot()
        if callable(qwen4_fp8_direct_snapshot)
        else (0, 0, 0, 0, 0, 0, 0, 0)
    )
    scale_snapshot = getattr(
        engine.store, "k3_scale_sidecar_snapshot", None
    )
    scale_stages = (
        scale_snapshot() if callable(scale_snapshot) else (0, 0, 0, 0)
    )
    nf12_snapshot = getattr(engine.store, "bf16_nf12_snapshot", None)
    nf12_stages = (
        nf12_snapshot() if callable(nf12_snapshot) else (0, 0, 0, 0)
    )
    parallel_snapshot = getattr(engine.store, "parallel_tier_snapshot", None)
    parallel_stages = (
        parallel_snapshot() if callable(parallel_snapshot) else (
            int(getattr(engine.store, "parallel_tier_fetches", 0) or 0),
            int(getattr(engine.store, "parallel_tier_fast_bytes", 0) or 0),
            int(getattr(engine.store, "parallel_tier_archive_bytes", 0) or 0),
            int(getattr(engine.store, "parallel_tier_wall_ns", 0) or 0),
            int(getattr(
                engine.store, "parallel_tier_fast_service_ns", 0) or 0),
            int(getattr(
                engine.store, "parallel_tier_archive_service_ns", 0) or 0),
            int(getattr(engine.store, "parallel_tier_hidden_ns", 0) or 0),
        )
    )
    return (
        int(stats.hits), int(stats.misses), int(stats.evictions),
        int(stats.bytes_read), int(getattr(stats, "pinned_hits", 0) or 0),
        int(getattr(stats, "prefetch_hits", 0) or 0),
        int(getattr(stats, "prefetch_waits", 0) or 0),
        int(float(getattr(stats, "prefetch_wait_s", 0.0) or 0.0) * 1e9),
        int(getattr(stats, "prefetch_loads", 0) or 0),
        int(getattr(stats, "prefetch_loaded_bytes", 0) or 0),
        int(getattr(stats, "prefetch_loaded_resident_bytes", 0) or 0),
        int(getattr(stats, "prefetch_oversize_pages", 0) or 0),
        int(float(getattr(stats, "prefetch_load_s", 0.0) or 0.0) * 1e9),
        int(getattr(stats, "prefetch_useful_pages", 0) or 0),
        int(getattr(stats, "prefetch_useful_bytes", 0) or 0),
        int(getattr(stats, "prefetch_useful_resident_bytes", 0) or 0),
        int(float(
            getattr(stats, "prefetch_useful_load_s", 0.0) or 0.0) * 1e9),
        int(getattr(stats, "prefetch_wasted_pages", 0) or 0),
        int(getattr(stats, "prefetch_wasted_bytes", 0) or 0),
        int(getattr(stats, "prefetch_wasted_resident_bytes", 0) or 0),
        int(float(
            getattr(stats, "prefetch_wasted_load_s", 0.0) or 0.0) * 1e9),
        int(engine.expert_hits),
        int(engine.expert_misses),
        int(getattr(governor, "reservations", 0) or 0),
        int(getattr(governor, "reservation_calls", 0) or 0),
        int(getattr(
            governor, "reservation_fast_path_calls", 0) or 0),
        int(getattr(
            governor, "reservation_clear_cache_only_calls", 0) or 0),
        int(getattr(
            governor, "reservation_reason_counts", {}
        ).get("serial-verify-layer-page", 0) or 0),
        int(getattr(
            governor, "reservation_reason_counts", {}
        ).get("serial-verify-transient", 0) or 0),
        int(getattr(
            governor, "reservation_reason_counts", {}
        ).get("qwen-prefill-layer-page", 0) or 0),
        int(getattr(
            governor, "reservation_reason_counts", {}
        ).get("qwen-prefill-transient", 0) or 0),
        int(getattr(
            governor, "reservation_reason_counts", {}
        ).get("glm53-expert-page", 0) or 0),
        int(sum(
            int(count or 0)
            for reason, count in getattr(
                governor, "reservation_reason_counts", {}
            ).items()
            if str(reason).startswith("glm53-")
            and str(reason) != "glm53-expert-page"
        )),
        int(getattr(
            governor, "reservation_requested_bytes", 0) or 0),
        int(getattr(
            governor, "reservation_budget_reduced_bytes", 0) or 0),
        int(getattr(
            governor, "reservation_budget_restored_bytes", 0) or 0),
        int(getattr(
            governor, "reservation_cache_released_bytes", 0) or 0),
        int(getattr(
            governor, "reservation_unproductive_shrinks", 0) or 0),
        int(getattr(
            governor, "reservation_zero_release_short_circuits", 0) or 0),
        int(getattr(governor, "reservation_failures", 0) or 0),
        int(getattr(governor, "swap_pressure_events", 0) or 0),
        int(getattr(
            governor, "swap_pressure_used_growth_bytes", 0) or 0),
        int(getattr(
            governor, "swap_pressure_out_growth_bytes", 0) or 0),
        int(getattr(engine.store, "fast_tier_bytes", 0) or 0),
        int(getattr(engine.store, "archive_bytes", 0) or 0),
        *parallel_stages,
        *store_stages,
        *glm53_fp8_stages,
        *glm53_fp8_direct_stages,
        *qwen4_fp8_direct_stages,
        *scale_stages,
        *nf12_stages,
    )


def _direct_io_snapshot(engine) -> dict[str, int] | None:
    """Snapshot direct-reader counters without requiring every backend.

    Several speculative adapters bypass :meth:`StreamingEngine.generate` and
    call the target's lower-level sweeps directly.  Keeping this small helper
    next to the cache-I/O accounting lets those adapters publish the same
    physical-read evidence as ordinary generation instead of silently losing
    it at the wrapper boundary.
    """
    store = getattr(engine, "store", None)
    snapshot = getattr(store, "direct_io_snapshot", None)
    if not callable(snapshot):
        return None
    return {str(key): int(value) for key, value in snapshot().items()}


def _record_direct_io_delta(
    engine, before: dict[str, int] | None, stats: dict, *,
    after: dict[str, int] | None = None,
) -> None:
    """Publish one request's direct-reader work and active policy state."""
    if before is None:
        return
    after = _direct_io_snapshot(engine) if after is None else after
    if after is None:
        return
    for key in (
        "fd_opens", "fd_hits", "fd_closes", "fd_open_ns",
        "fd_nocache_applied", "pread_calls", "pread_requested_bytes",
        "pread_bytes", "pread_ns", "pread_short_reads",
    ):
        stats[f"direct_io_{key}"] = max(
            0, int(after.get(key, 0)) - int(before.get(key, 0)))
    stats["direct_io_fd_cached"] = int(after.get("fd_cached", 0))
    stats["direct_io_fd_cache_enabled"] = int(
        after.get("fd_cache_enabled", 0))
    stats["direct_io_nocache_enabled"] = int(
        getattr(getattr(engine, "store", None), "_direct_fd_nocache", False))


def _record_cache_io_delta(
    engine, before: tuple[int, ...], stats: dict, *,
    prefix: str = "", after: tuple[int, ...] | None = None,
) -> None:
    """Expose cache/I/O evidence without confusing cumulative engine totals."""
    after = _cache_io_snapshot(engine) if after is None else after
    keys = (
        "weight_cache_hits", "weight_cache_misses",
        "weight_cache_evictions", "weight_store_bytes_read",
        "weight_cache_pinned_hits", "weight_cache_prefetch_hits",
        "weight_prefetch_waits", "weight_prefetch_wait_ns",
        "weight_prefetch_loads", "weight_prefetch_loaded_bytes",
        "weight_prefetch_loaded_resident_bytes",
        "weight_prefetch_oversize_pages",
        "weight_prefetch_load_ns", "weight_prefetch_useful_pages",
        "weight_prefetch_useful_bytes",
        "weight_prefetch_useful_resident_bytes",
        "weight_prefetch_useful_load_ns",
        "weight_prefetch_wasted_pages", "weight_prefetch_wasted_bytes",
        "weight_prefetch_wasted_resident_bytes",
        "weight_prefetch_wasted_load_ns",
        "expert_cache_hits", "expert_cache_misses",
        "governor_reservations", "governor_reservation_calls",
        "governor_reservation_fast_path_calls",
        "governor_reservation_clear_cache_only_calls",
        "governor_serial_verify_page_reservation_calls",
        "governor_serial_verify_transient_reservation_calls",
        "governor_qwen_prefill_page_reservation_calls",
        "governor_qwen_prefill_transient_reservation_calls",
        "governor_glm53_expert_page_reservation_calls",
        "governor_glm53_transient_reservation_calls",
        "governor_reservation_requested_bytes",
        "governor_reservation_budget_reduced_bytes",
        "governor_reservation_budget_restored_bytes",
        "governor_reservation_cache_released_bytes",
        "governor_reservation_unproductive_shrinks",
        "governor_reservation_zero_release_short_circuits",
        "governor_reservation_failures",
        "governor_swap_pressure_events",
        "governor_swap_used_growth_bytes",
        "governor_swap_out_growth_bytes",
        "weight_fast_tier_bytes", "weight_archive_bytes",
        "parallel_tier_fetches", "parallel_tier_fast_bytes",
        "parallel_tier_archive_bytes", "parallel_tier_wall_ns",
        "parallel_tier_fast_service_ns",
        "parallel_tier_archive_service_ns", "parallel_tier_hidden_ns",
        "ct_mxfp4_transform_ns", "ct_mxfp4_transform_calls",
        "ct_mxfp4_input_bytes", "ct_mxfp4_resident_bytes",
        "glm53_fp8_transform_ns", "glm53_fp8_transform_calls",
        "glm53_fp8_native_calls", "glm53_fp8_input_bytes",
        "glm53_fp8_resident_bytes",
        "glm53_fp8_prefetch_transform_ns",
        "glm53_fp8_prefetch_transform_calls",
        "glm53_fp8_prefetch_native_calls",
        "glm53_fp8_direct_pages",
        "glm53_fp8_direct_resident_bytes",
        "glm53_fp8_direct_qmv_calls",
        "glm53_fp8_direct_qmv_positions",
        "glm53_fp8_direct_fallback_calls",
        "glm53_fp8_direct_fallback_positions",
        "glm53_fp8_direct_fallback_reconstruct_ns",
        "glm53_fp8_direct_fallback_reconstruct_bytes",
        "qwen4_fp8_direct_pages",
        "qwen4_fp8_direct_resident_bytes",
        "qwen4_fp8_direct_qmv_calls",
        "qwen4_fp8_direct_qmv_positions",
        "qwen4_fp8_direct_fallback_calls",
        "qwen4_fp8_direct_fallback_positions",
        "qwen4_fp8_direct_fallback_reconstruct_ns",
        "qwen4_fp8_direct_fallback_reconstruct_bytes",
        "k3_scale_sidecar_read_bytes", "k3_scale_sidecar_output_bytes",
        "k3_scale_sidecar_decode_ns", "k3_scale_sidecar_decode_calls",
        "bf16_nf12_read_bytes", "bf16_nf12_output_bytes",
        "bf16_nf12_decode_ns", "bf16_nf12_decode_calls",
    )
    for key, start, end in zip(keys, before, after, strict=True):
        stats[prefix + key] = max(0, end - start)
    if not prefix:
        stats["weight_cache_resident_bytes"] = int(engine.cache.total_bytes)
        stats["weight_cache_budget_bytes"] = int(engine.cache.max_bytes)
        stats["weight_cache_pinned_bytes"] = int(getattr(
            engine.cache, "pinned_bytes", 0) or 0)
        stats["weight_cache_prefetched_bytes"] = int(getattr(
            engine.cache, "prefetched_bytes", 0) or 0)
        stats["planned_trunk_pin_layers"] = int(getattr(
            engine, "planned_trunk_pin_layers", 0) or 0)
        stats["planned_trunk_pin_bytes"] = int(getattr(
            engine, "planned_trunk_pin_bytes", 0) or 0)
        stats["weight_prefetch_depth"] = int(getattr(
            getattr(engine, "rc", None), "prefetch_depth", 0) or 0)
        stats["layer_transient_bytes"] = int(engine._layer_transient)
        stats["prefill_layer_transient_bytes"] = int(getattr(
            engine, "_prefill_layer_transient", 0) or 0)
        stats["decode_layer_transient_bytes"] = int(getattr(
            engine, "_decode_layer_transient", 0) or 0)
        stats["layer_transient_margin_bytes"] = int(getattr(
            engine, "_layer_transient_margin", 400_000_000))
        stats["token_transient_bytes"] = int(engine._token_transient)
    stats[prefix + "weight_prefetch_wait_s"] = (
        stats[prefix + "weight_prefetch_wait_ns"] / 1e9)
    stats[prefix + "weight_prefetch_load_s"] = (
        stats[prefix + "weight_prefetch_load_ns"] / 1e9)
    stats[prefix + "weight_prefetch_useful_load_s"] = (
        stats[prefix + "weight_prefetch_useful_load_ns"] / 1e9)
    stats[prefix + "weight_prefetch_wasted_load_s"] = (
        stats[prefix + "weight_prefetch_wasted_load_ns"] / 1e9)
    stats[prefix + "weight_prefetch_hidden_lower_bound_s"] = max(
        0.0,
        stats[prefix + "weight_prefetch_useful_load_s"]
        - stats[prefix + "weight_prefetch_wait_s"],
    )
    stats[prefix + "parallel_tier_wall_s"] = (
        stats[prefix + "parallel_tier_wall_ns"] / 1e9)
    stats[prefix + "parallel_tier_fast_service_s"] = (
        stats[prefix + "parallel_tier_fast_service_ns"] / 1e9)
    stats[prefix + "parallel_tier_archive_service_s"] = (
        stats[prefix + "parallel_tier_archive_service_ns"] / 1e9)
    stats[prefix + "parallel_tier_hidden_s"] = (
        stats[prefix + "parallel_tier_hidden_ns"] / 1e9)


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
        if rc.rerank_lm_head_source_fingerprint:
            identity += (
                "+rowpaged-"
                + rc.rerank_lm_head_source_fingerprint[:16])
    if rc.resident_attention_mode:
        identity += (
            f"+residentattn-{rc.resident_attention_mode}"
            f"q{rc.resident_attention_bits}"
            f"g{rc.resident_attention_group_size}"
        )
    if rc.expert_top_k_by_layer:
        identity += "+olmoe-topk-" + ".".join(
            str(top_k) for top_k in rc.expert_top_k_by_layer)
    if getattr(rc, "native_ct_mxfp4", False):
        identity += "+ct-mxfp4-native"
    if getattr(store, "native_glm53_fp8_dequant", False):
        identity += "+glm53-fp8-dequant-native"
        if not getattr(store, "native_glm53_fp8_prefetch", True):
            identity += "-foreground-only"
    if getattr(store, "glm53_fp8_direct_qmv", False):
        identity += "+glm53-fp8-direct-qmv"
        if getattr(store, "glm53_fp8_direct_qmv_decode_only", False):
            identity += "-decode-only"
    if getattr(store, "qwen4_fp8_direct_qmv", False):
        identity += "+qwen4-fp8-direct-qmv"
        if getattr(store, "qwen4_fp8_direct_qmv_decode_only", False):
            identity += "-decode-only"
    return identity


def expert_route_overlap_summary(
    positions_by_expert: dict[int, list[int]],
    previous_route: tuple[int, ...] | None = None,
) -> tuple[dict[str, int], tuple[int, ...]]:
    """Summarize route union growth and adjacent-position reuse.

    The input already exists after authoritative routing. No expert is
    predicted and no model decision changes. ``previous_route`` is the last
    committed position for this same layer from the prior sweep, so the
    boundary pair is the exact quantity a one-token warm expert cache could
    reuse.
    """
    routes: dict[int, set[int]] = {}
    for expert, positions in positions_by_expert.items():
        for position in positions:
            routes.setdefault(int(position), set()).add(int(expert))
    ordered = [routes[position] for position in sorted(routes)]
    if not ordered:
        raise ValueError("expert route overlap requires at least one position")

    union = set().union(*ordered)
    selected_slots = sum(len(route) for route in ordered)
    adjacent_intersection = 0
    adjacent_union = 0
    adjacent_current = 0
    exact_pairs = 0
    within_pairs = 0
    cross_pairs = 0
    cross_intersection = 0
    cross_current = 0

    previous = set(previous_route) if previous_route is not None else None
    for index, current in enumerate(ordered):
        if previous is not None:
            intersection = len(previous & current)
            adjacent_intersection += intersection
            adjacent_union += len(previous | current)
            adjacent_current += len(current)
            exact_pairs += int(previous == current)
            if index == 0:
                cross_pairs = 1
                cross_intersection = intersection
                cross_current = len(current)
            else:
                within_pairs += 1
        previous = current

    summary = {
        "calls": 1,
        "positions": len(ordered),
        "selected_slots": selected_slots,
        "union_experts": len(union),
        "adjacent_pairs": within_pairs + cross_pairs,
        "within_call_pairs": within_pairs,
        "cross_call_pairs": cross_pairs,
        "adjacent_intersection_experts": adjacent_intersection,
        "adjacent_union_experts": adjacent_union,
        "adjacent_current_experts": adjacent_current,
        "exact_adjacent_pairs": exact_pairs,
        "cross_call_intersection_experts": cross_intersection,
        "cross_call_current_experts": cross_current,
    }
    return summary, tuple(sorted(ordered[-1]))


def offset_expert_route_positions(
    positions_by_expert: dict[int, list[int]] | None,
    position_offset: int,
) -> dict[int, list[int]] | None:
    """Move slice-local router rows into their enclosing verifier window."""
    if positions_by_expert is None:
        return None
    offset = int(position_offset)
    return {
        int(expert): [
            offset + int(position)
            for position in positions
        ]
        for expert, positions in positions_by_expert.items()
    }


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
    # Optional hard Metal ceiling for explicitly memory-bounded profiles.
    # Zero preserves the device-recommended default used by existing models.
    # Qwen4's 49K host-spooled prefill sets this to 8.5 GB and also checks the
    # boundary synchronously at its tile materialization points.
    metal_limit_mb: int = 0
    # Exact Qwen4 PLE row reads can issue one tiny random extent per hashed
    # n-gram row. One preserves the baseline; explicit profiles may overlap
    # independent preads while retaining identical BF16 row bytes/order.
    qwen4_ple_read_workers: int = 1
    # Candidate layer-stationary MoE schedule. Gather only the prompt rows
    # assigned to each expert and evaluate that expert once per layer instead
    # of re-uploading every mixed tile for every expert-page batch. The served
    # target weights remain released BF16; default-off until real checkpoint
    # row-shape/state oracles prove the changed batch geometry bit-identical.
    qwen4_global_expert_rows: bool = False
    # Exact-shape candidate: for each existing tile/expert-page batch, upload
    # only the union of positions routed to that batch. Every individual
    # expert still receives the same rows, order, shape, and accumulation.
    qwen4_sparse_expert_batch_rows: bool = False
    # Evaluate this many already-independent routed tile accumulators in one
    # mx.eval call. Each expert GEMM retains its original per-tile shape and
    # every tile retains ascending expert accumulation; only the host/device
    # synchronization boundary is coalesced. One is the validated baseline.
    qwen4_expert_tile_eval_batch: int = 1
    # When a Qwen4 exact fast tier is configured, keep the long host-spooled
    # prefill on the archive device only and enable the two-device overlay for
    # decode. This avoids competing OS read caches at the 49K memory peak.
    qwen4_fast_tier_decode_only: bool = False
    # Exact I/O-only overlap for Qwen4's long prefill. Routed expert batches
    # are fetched on the existing bounded worker during prefill, then decode
    # returns to the validated synchronous schedule. This avoids carrying the
    # measured decode/swap regression of globally enabled expert prefetch.
    qwen4_expert_batch_prefetch_prefill_only: bool = False
    # Keep the exact untied Qwen4 output head absent during host-spooled
    # prefill, then load and pin it at the first post-prefill projection.  The
    # next request releases it before touching the target trunk.  This keeps
    # the 1.27 GB released BF16 head out of the 49K prompt high-water while
    # avoiding repeated row streaming during target-verified MTP decode.
    qwen4_phase_lm_head: bool = False
    # Exact, default-off extension of the phase-scoped Qwen4 head lifetime.
    # Native MTP has already materialized every draft probability before its
    # authoritative multi-position target sweep starts, so the 1.27-GB BF16
    # head can be released for that trunk sweep and demand-loaded exactly once
    # at the verifier's final projection.  This changes storage lifetime only;
    # the target weights, operator order, and logits remain released-model
    # exact.  Keep it separate from qwen4_phase_lm_head until the live pressure
    # and heterogeneous-replay gates clear.
    qwen4_serial_verify_suspend_lm_head: bool = False
    # Use the released-BF16 singleton-equivalent Metal GEMV for independent
    # Qwen4 verifier MLP/router rows. Default-off until real checkpoint state,
    # token, and heterogeneous-request gates clear.
    qwen4_serial_verify_exact_bf16_gemv: bool = False
    # Reuse immutable normalized/RoPE'd QSA pooled index keys while serial
    # verifier positions remain inside the same four-token compression block.
    # Raw released-BF16 QSA keys remain authoritative; trims invalidate the
    # derived cache. Default-off pending real long-context equality/timing.
    qwen4_qsa_pool_cache: bool = False
    # Opt-in request-local attribution. "" disables it; "layers" records the
    # runtime's existing materialization boundaries; "ops" adds diagnostic
    # attention/router/MLP barriers for supported Qwen/Kimi/GLM hybrid blocks.
    execution_profile: str = ""
    # Opt-in, lossless representation path for published compressed-tensors
    # MXFP4 weights. It retains the released E2M1/E8M0 bytes and feeds them
    # directly to MLX's native packed matmul instead of eagerly expanding BF16.
    native_ct_mxfp4: bool = False
    # Explicit lossless Kimi K3 E8M0 scale overlay. Empty disables it. The
    # immutable sidecar generation is checkpoint-fingerprinted and may cover a
    # subset of layers; uncovered layers use released safetensors unchanged.
    kimi_k3_scale_sidecar_dir: str = ""
    # Explicit exact BF16 trunk representation. Empty disables it; partial
    # generations fall back to the released safetensors tensors.
    bf16_nf12_sidecar_dir: str = ""
    # Descriptor-level Darwin F_NOCACHE reads for the exact NF12 stream.
    # Avoids one-shot compressed pages competing with live Metal allocations.
    bf16_nf12_uncached_reads: bool = False
    # Consume eligible exact NF12 rank-2 operands inside a small-M linear
    # kernel instead of first materializing dense BF16 matrices.
    bf16_nf12_direct_linear: bool = False
    # Explicit raw-safetensors read-order experiment. Sort each requested
    # shard group by immutable payload offset before MLX evaluation.
    safetensors_offset_order: bool = False
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
    # F197: derive the pinned trunk prefix from a byte budget instead of a hand
    # -chosen layer count. A trunk is read strictly cyclically, which defeats
    # recency/frequency eviction completely (0% hits at any sub-trunk budget,
    # proven against the real cache in tests/test_f197_pinned_trunk_prefix.py),
    # so budget spent on trunk residency only pays when it is pinned. Zero
    # keeps the explicit pin_first_layers count in charge.
    pin_trunk_budget_mb: int = 0
    # Capacity pin planning must leave for routed expert pages. Only consulted
    # when pin_trunk_budget_mb is set.
    pin_trunk_expert_reserve_mb: int = 0
    prefetch_depth: int = 0  # 0 disables prefetch
    prefetch_workers: int = 0  # 0 = store default (raw: 1, packed: 2)
    max_kv_mb: int = 0  # 0 = unpaged KV (all resident); >0 enables disk spilling
    # Explicit lossy-Qwen sidequest: one-token paged attention uses a fused
    # tile-wise online softmax instead of materializing the complete history.
    # The changed reduction order is never enabled on released/lossless paths.
    qwen35_paged_online_attention: bool = False
    qwen35_paged_online_tile_positions: int = 2048
    qwen35_paged_online_page_native: bool = False
    # Explicit Qwen hybrid mode: durable journal tensors restore directly into
    # bounded PagedKVCache pages and are never preloaded into the resident LRU.
    # Default-off until the real 49K replay passes the cold/restart proof gate.
    paged_kv_persist: bool = False
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
    rerank_lm_head_source: str = ""  # verified released BF16 checkpoint;
    # row reads only, never pinned as one resident exact tensor
    rerank_lm_head_source_fingerprint: str = ""
    # Full exact-head scans used to measure actual shortlist recall. Disabled
    # by default because each probe deliberately reads the complete 2.543 GB
    # source head; N means every Nth reranked projection.
    rerank_lm_head_recall_probe_every: int = 0
    # Explicit privacy-safe promotion evidence. Each authoritative target
    # projection is reduced to the exact winner's approximate rank plus two
    # booleans; prompts, hidden states, logits, token IDs, and text never land
    # on disk. Hard bounds and the offline 1000-position/100% gate live in
    # runtime.lm_head_recall_capture. Empty keeps the path fully disabled.
    rerank_lm_head_rank_capture_path: str = ""
    rerank_lm_head_rank_capture_max_positions: int = 1200
    rerank_lm_head_rank_capture_max_per_request: int = 128
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
    parallel_storage_reads: bool = False  # overlap authenticated fast-overlay
    # decode with archive reads only when the paths resolve to distinct devices
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
    # F94: Qwen3.5/3.6's native single-depth MTP (real mtp.* checkpoint
    # weights) as a verified draft source, wired via
    # runtime.qwen35_mtp.QwenMTPSpeculativeEngine (a server.py construction
    # decision, not consumed inside StreamingEngine itself -- this flag just
    # signals intent/opt-in the same way draft_dir presence gates the
    # existing SpeculativeEngine wrapper).
    qwen_mtp_speculative: bool = False
    # SQ26: zmlx's fused DeltaNet decode kernels (fused_conv1d_silu,
    # gated_rmsnorm_silu). Real, measured decode-shape (L=1) speedup on this
    # hardware (1.81x/1.38x); a genuine bf16 precision difference exists in
    # isolation (0.03125/0.0625 max abs diff vs this codebase's own
    # float32-accumulated implementations) but a real greedy-token quality
    # gate against Qwen3.5-4B (2026-07-23) found zero divergence over 24
    # tokens -- see tests/test_zmlx_fused_deltanet_decode.py. Gated to
    # fast/lossy mode only in server.py, never lossless (the precision
    # difference is real even if it happens not to flip any observed
    # argmax); applied only for L==1 decode shape, never prefill.
    zmlx_fused_deltanet_decode: bool = False
    # F103: from-scratch mx.fast.metal_kernel fusion of the single-step
    # gated-delta-rule recurrence body (decay-scale, predicted dot, delta,
    # state update, output dot) into one Metal dispatch, decode (L==1) only.
    # Distinct from zmlx_fused_deltanet_decode above (a third-party library,
    # SQ26, found to be a net decode slowdown despite an isolated-benchmark
    # win) -- this is this project's own kernel, written to test whether a
    # tightly-scoped, hand-written fusion avoids zmlx's per-call dispatch
    # overhead. See tests/test_native_fused_deltanet_decode.py for the
    # correctness gate and STATUS.md for whichever real end-to-end verdict
    # that test's A/B produced -- an isolated microbenchmark win is
    # deliberately NOT trusted on its own given the zmlx precedent.
    native_fused_deltanet_decode: bool = False
    # Opt-in multi-position Kimi KDA recurrence fusion. This keeps the
    # released serial state equation but performs its FP32 reductions inside
    # one Metal kernel, so it is algebraically exact while not necessarily
    # activation-byte-identical to MLX's separately dispatched reductions.
    # Keep disabled until real K3 greedy and continuation gates admit it.
    kimi_k3_native_fused_kda_prefill: bool = False
    # Opt-in byte-identical KDA prefill graph compilation. This retains the
    # ordinary MLX reduction operators and 32-position state boundaries while
    # amortizing graph dispatch/optimization overhead.
    kimi_k3_compiled_kda_prefill: bool = False
    # Opt-in byte-identical Qwen DeltaNet prefill graph compilation. Like the
    # K3 path above, this traces the ordinary recurrence operators in bounded
    # 32-position segments without the WY path's FP32 reassociation.
    qwen_compiled_delta_prefill: bool = False
    # Explicit serial Metal scan. The current 128x128 implementation mirrors
    # MLX 0.32's BM=32 strided-reduction association and disables contraction;
    # direct and real-checkpoint Qwen4 state oracles are byte-identical. The
    # Qwen3.5 server route remains classified lossy until its own heterogeneous
    # real-model corpus is rerun under this replacement kernel.
    qwen_native_fused_delta_prefill: bool = False
    # Chunkwise WY DeltaNet prefill. Numerically close but not
    # activation-identical across arbitrary checkpoint splits, so server.py
    # admits it automatically only for fast/lossy Qwen3.5/3.6 routes.
    qwen_chunked_delta_prefill: bool = False
    # 2026-07-27: fp8 (e4m3) storage for Qwen3.5/3.6's ordinary full-attention
    # KV cache (never the DeltaNet/KDA recurrent state -- that's small and
    # fixed-size regardless of context length). Genuinely lossy; explicit
    # opt-in only (VMODEL_QWEN35_FP8_KV_CACHE=1), no auto-default, per
    # CLAUDE.md/AGENTS.md's "Avoiding overfit defaults" rule -- this has not
    # been validated broadly enough to default on. See
    # tests/test_qwen35_oracle.py's fp8 case for the measured precision cost.
    qwen_fp8_kv_cache: bool = False
    # F11: prompt-lookup (n-gram) speculative decoding, mutually exclusive
    # with qwen_mtp_speculative above for this first version (see
    # EngineManager.get()'s construction site in server.py). Zero-model --
    # proposals come from repeated substrings already in the token history
    # (runtime/speculative.py's ngram_propose), free when there's no match,
    # and can propose k>1 tokens per round unlike MTP's fixed k=1. Real
    # greedy-token quality gate: tests/test_qwen35_ngram_speculative.py.
    qwen_ngram_speculative: bool = False
    # Grammar fast-forward (2026-07-23, token-level jump-forward decoding):
    # under constrained decoding, whenever the grammar allows exactly ONE
    # legal next token, masked argmax is forced regardless of model logits,
    # so the token needs no per-token model sweep -- only a KV/state update,
    # which generate() batches into one multi-position feed. Byte-identical
    # to the plain constrained path by construction (token-level, NOT
    # SGLang's string-level variant, which can change tokenization). Targets
    # exactly the workload every speculative scheme above failed on: forced
    # tool-call JSON structure under a grammar constraint.
    grammar_fast_forward: bool = False
    # String-level jump-forward (SGLang-style), fast/lossy profile ONLY:
    # commits the canonical tokenization of grammar-forced TEXT spans via
    # matcher.find_jump_forward_string(). Rendered text stays grammar-forced
    # but token ids can differ from what per-token masked argmax would have
    # picked (multiple legal tokenizations), so this is never enabled for
    # the lossless target -- see GrammarConstraint.forced_run's docstring
    # for the measured motivation (token-level forcing almost never fires).
    grammar_jump_forward_lossy: bool = False
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
    hot_prompt_kv_persist_max_mb: int = 0  # 0 = checkpoint-count limit only;
    # otherwise GC also bounds all reachable immutable segment/checkpoint
    # bytes. This is especially important for long K3 MLA prefix snapshots.
    # Side-quest-only override for a Qwen2 checkpoint that does not itself
    # declare rope_scaling. 0/1 = released RoPE; >1 = static YaRN extrapolation.
    qwen_yarn_factor: float = 0.0
    prefill_chunk_size: int = 0  # bound prefill compute/transient memory WITHOUT writing state
    # Explicit Qwen3.5/Qwen3.8 hybrid safety cap applied after the live
    # per-conversation choice and any memory-retry cap. 0 preserves the
    # automatic ladder unchanged. Unlike prefill_chunk_size, this is a stable
    # operator/profile policy and is never rewritten by a request.
    qwen35_prefill_chunk_ceiling: int = 0
    # Explicit verifier-memory experiment.  Standard MLX QTensor pages have an
    # exact metadata-derived payload estimate plus a 5% pad, so a caller may
    # reuse the selected one-position compute margin instead of reserve()'s
    # generic 400-MB unknown-allocation margin.  Default false preserves the
    # conservative historical admission policy.
    qwen35_serial_verify_exact_page_admission: bool = False
    # Explicit fast-target experiment: keep attention/DeltaNet recurrence in
    # canonical position order, but evaluate the dense, position-independent
    # SwiGLU residual for the complete verifier window in one batched call.
    # The served BF16 lossless route never enables this; MXFP4 promotion still
    # requires identical target tokens plus the heterogeneous quality corpus.
    qwen35_serial_verify_batched_mlp: bool = False
    # Exact-target, default-off memory-lifetime optimization for an untied
    # Qwen LM head. Startup registers an exact dormant lease instead of
    # materializing the head before prefill. The head is pinned on its first
    # projection, released before each multi-position target-trunk sweep, then
    # the verifier's demand-loaded head is re-pinned without a second read.
    qwen35_serial_verify_suspend_lm_head: bool = False
    # Content-blind activation boundary for the explicit head lifecycle.
    # 8192 is the measured production candidate. Lower values are useful only
    # for composed experiments (for example a wider exact verifier that needs
    # the physical head release to fit) and must retain their own live gate.
    qwen35_serial_verify_suspend_lm_head_min_prompt_tokens: int = (
        QWEN35_PHASE_HEAD_MIN_PROMPT_TOKENS)
    # F94: layer-major (not chunk-major) dense prefill for qwen3_5 (dense
    # hybrid DeltaNet/full-attention, e.g. Qwen3.5-4B/9B, Qwen3.6-27B) --
    # fetches each layer's weights exactly once for the whole prefill instead
    # of once per chunk. Opt-in only for a first live rollout; see
    # StreamingEngine._layer_stationary_qwen35_sweep and CLAUDE.md's
    # 2026-07-23 note on why chunk-major re-reads dominate prefill time here.
    layer_stationary_prefill: bool = False
    # Exact Qwen-only extension of layer-stationary prefill: capture the
    # stable chat endpoint inside the same per-layer pass that consumes the
    # trailing generation scaffold. Explicit opt-in until broad real-shape
    # replay proves the new schedule beyond its focused oracle.
    qwen_fused_boundary_scaffold_prefill: bool = False
    # Positions per MoE tile inside a layer-stationary DeepSeek V4 prefill.
    # Bounds the float32 hyper-connection carrier without costing weight
    # reads: a layer's routed experts stay resident across its own tiles.
    dsv4_ffn_tile_width: int = 2048
    # Positions per DeepSeek V4 prefix checkpoint. A harness prompt is almost
    # entirely a stable prefix -- the captured 51,220-token request is 46,941
    # tokens of tool schemas plus 4,233 of system prompt against a 42-token
    # user message -- so resuming from a checkpoint at the largest stride
    # boundary below the divergence skips nearly all of prefill on every turn
    # after the first. 0 disables.
    dsv4_prefix_checkpoint_stride: int = 8192
    # Directory for on-disk prefix checkpoints. Lets a COLD start skip
    # the tool-preamble prefill entirely, since that preamble is known
    # before any request arrives.
    dsv4_prefix_cache_dir: str = ""
    # Default ON. Only gpt-oss consults it (see the model_type guard at the
    # use site); it drops keys sliding layers provably cannot read.
    gptoss_sliding_kv_window: bool = True
    # Explicit lossy Qwen hybrid prefill schedule. The first N layers consume
    # the full prompt; a fixed P-position prefix anchor plus the final S hidden
    # positions continue through the remaining layers. P=0 preserves the
    # original suffix-only experiment. Zeroes preserve the released full-depth
    # computation. This is request-content independent and is admitted only by
    # the named fast-profile environment opt-in in server.py.
    qwen_lossy_suffix_prefill_early_layers: int = 0
    qwen_lossy_suffix_prefill_prefix_tokens: int = 0
    qwen_lossy_suffix_prefill_tokens: int = 0
    # Default-off durable exact-prompt endpoint for the mixed-depth sidequest
    # journal. Stable pre-user boundaries remain independently enabled.
    qwen_mixed_depth_endpoint_persist: bool = False
    # Optional system-available floor enforced after a mixed-depth Qwen
    # response by shedding consumed LRU weight pages. Kept separate from the
    # pre-allocation/hot-KV admission floor so it cannot perturb the arithmetic
    # or cache residency of the request being measured.
    qwen_postgen_min_available_mb: int = 0
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
    adaptive_chunk_escalate_growth_cap: bool = False  # opt-in, default matches prior
    # behavior exactly when False. See AdaptiveChunkController's own docstring/comment:
    # escalates the per-step growth cap (2x -> up to 8x) after two consecutive
    # ceiling-clamped GREEN proposals, resetting to 2x immediately on any bad event.
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
    # K3-specific activation remains separate and default-off until the real
    # long-context gates clear.  The implementation itself is generic MLA
    # algebra: no prompt, tool, subject, route, or request-shape branch enters
    # either decision.
    glm53_sparse_absorbed_mla: bool = False
    # Fused selected-row K/V attention removes the gather but uses online
    # softmax/SIMD reductions, so it is an explicit lossy Metal candidate.
    glm53_sparse_fused_attention: bool = False
    # Optional compact prefill-only K/V for the already-lossy fused sparse
    # attention candidate. Per-head/per-row symmetric int8 scales halve the
    # expanded DSA cache; the released target weights and recurrent state stay
    # unchanged. Never enabled by a lossless/default profile.
    glm53_sparse_fused_kv_int8: bool = False
    # Gather every row routed to one expert across layer-stationary prefill
    # tiles before that expert's three GEMMs. Routing and ascending expert
    # accumulation stay unchanged, but the GEMM outer shape changes, so this
    # remains an explicit lossy candidate.
    glm53_coalesced_expert_positions: bool = False
    # Bound the gathered outer dimension for one coalesced expert GEMM. The
    # expert page remains resident across chunks, so this limits Metal scratch
    # without re-reading weights. Only used by the explicit lossy candidate.
    glm53_coalesced_expert_max_positions: int = 512
    # Reuse immutable completed GLM-5.3 DSA pool keys instead of rebuilding
    # the whole prefix for every 32-position tile. Explicit until the real
    # Metal greedy/state oracle confirms that outer-shape changes do not alter
    # any released BF16 result.
    glm53_incremental_dsa_pool: bool = False
    # Compile the ordinary GLM-5.3 KDA recurrence in bounded 32-position
    # segments. The implementation retains the reference MLX operators,
    # reduction order, and state materialization cadence; explicit until a
    # real checkpoint timing/token gate proves useful performance.
    glm53_compiled_kda_prefill: bool = False
    # Positions per compiled recurrence graph. Segment 32 is the measured
    # baseline; 16 is an explicit lower-peak candidate until a real-model gate.
    glm53_compiled_kda_segment: int = 32
    # Explicit lossy GLM-5.3 KDA prefill scan.  The fused Metal kernel keeps
    # the released causal recurrence and FP32 storage, but its in-kernel
    # reductions can associate differently from MLX's operator graph.  It is
    # therefore never selected by a lossless/default profile even when its
    # numerical error is far below the greedy-token gate.
    glm53_native_fused_kda_prefill: bool = False
    # Exact-capacity candidate for GLM-5.3 layer-stationary activations.
    # Materialized BF16 activations are copied to host as raw uint16 payloads
    # at explicitly bounded phase boundaries, then restored at the original
    # operator shapes. No floating-point conversion; default-off until real
    # output/state/read gates prove the full and Flash compositions independently.
    glm53_layer_stationary_host_spool: bool = False
    # Exact F75 query/key tiling for the official glm_moe_dsa indexer. The
    # score reduction dimension is unchanged; only the bounded candidate merge
    # schedule differs. It is inert below index_topk and remains explicit at
    # the server until the real long-context conformance gate passes.
    glm_dsa_key_tile_size: int = 256
    # Exact spare-capacity rows for the full GLM index-key history. Zero keeps
    # the legacy concat path; the explicit long-context route uses 1024.
    glm_dsa_index_step_size: int = 0
    # Allocate every full indexer's final stepped prompt capacity on its first
    # tile. This is content-independent and byte-exact, but remains explicit
    # until the real long-context pressure/timing gate proves the allocation
    # schedule is beneficial on the 16-GB host.
    glm_dsa_index_preallocate: bool = False
    # Query rows scored together by the exact full-GLM DSA selector. Compact
    # attention retains prefill_chunk_size; this independent width amortizes
    # score/merge launches without expanding more selected MLA rows at once.
    # Zero preserves the established coupled path.
    glm_dsa_selection_query_tile_size: int = 0
    # Position rows per dense-MLP materialization in full GLM's layer-major
    # prefill. The first three released dense layers otherwise form gate/up
    # intermediates across the complete prompt (multiple GB at 46.8K). Rows
    # are independent and retain their hidden-dimension reduction order.
    # Zero preserves the historical whole-prompt call shape.
    glm_dsa_dense_mlp_tile_size: int = 0
    # Algebraically absorb full GLM's released kv_b projection around dense
    # and query-specific selected-row MLA prefill. This avoids expanding
    # K selected latents into 64-head K/V per query, but changes floating
    # association and therefore remains an explicit greedy-gated candidate.
    glm_dsa_sparse_absorbed_mla: bool = False
    # External-volume scratch for exact released-dtype compressed MLA rows of
    # full GLM-5.3. Completed layer state is dead during later prefill layers,
    # then restored one layer at a time for decode.
    glm_dsa_mla_kv_spill_dir: str = ""
    # Official sparse-MLA serving layout for GLM-5.3. Algebraically folds the
    # released kv_b projection around attention instead of expanding 2,048
    # selected latents into 64-head K/V separately for every query. Explicit
    # opt-in because the reassociated floating reductions need real greedy
    # corpus gates even though the target weights and selected rows are exact.
    kimi_k3_compressed_mla: bool = False
    kimi_k3_absorbed_mla: bool = False
    # Maximum cached latent rows in one online-softmax score tile.  This value
    # is inert unless kimi_k3_absorbed_mla is explicitly enabled.
    kimi_k3_mla_key_tile_size: int = 2048
    # Fused Metal AttnRes readout, evaluated in bounded position tiles. Zero
    # retains the released composite MLX path. Explicit opt-in pending a full
    # real-model greedy gate and broader prompt-shape corpus.
    kimi_k3_fused_attnres_tile_size: int = 0
    # Optional external-volume scratch for exact BF16 AttnRes snapshots. This
    # is a memory-tiering choice, not compression or quantization; it requires
    # the fused tiled readout so snapshots are read back in bounded row tiles.
    kimi_k3_attnres_spill_dir: str = ""
    # Exact FP32/BF16 recurrent KDA endpoints can be tiered after each
    # completed prefill layer and lazily restored for decode. This bounds the
    # otherwise depth-growing ~6.7-MB-per-KDA-layer Metal residency.
    kimi_k3_kda_spill_dir: str = ""
    # Exact released-dtype compressed MLA latents are likewise dead after
    # their layer's prefill. Tier them and restore each layer lazily for
    # decode, bounding the ~73-MB-per-full-attention-layer resident growth.
    kimi_k3_mla_kv_spill_dir: str = ""
    # Bound K3's 33,792-wide dense MLP gate/up activations during a
    # layer-stationary long prefill. Zero preserves the existing full-position
    # call; explicit for the same rollout/generality reasons as fused AttnRes.
    kimi_k3_dense_mlp_tile_size: int = 0
    # Optional request-length scheduler for K3 prefill. ``fixed`` preserves
    # the two legacy settings above/at ``prefill_chunk_size``. The explicit
    # ``prompt-length`` policy selects a separately bounded short schedule
    # below one rendered-token threshold and those legacy values above it.
    # This policy is content-blind and remains opt-in through a named profile.
    kimi_k3_prefill_tile_policy: str = "fixed"
    kimi_k3_prefill_long_context_tokens: int = 256
    kimi_k3_prefill_short_tile_width: int = 256
    kimi_k3_dense_mlp_short_tile_size: int = 0
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
    expert_compute_batch: int = 0  # optional arithmetic/materialization boundary
    # inside an expert I/O batch. 0 preserves the historical coupled behavior.
    # A smaller positive value lets storage fetch/coalesce more pages while
    # retaining the already-validated floating-point accumulation grouping.
    decode_expert_fetch_batch: int = 0  # optional larger batch when routing covers
    # exactly one position; unlike prefill, decode's union is bounded by top-k
    # Lossless one-batch I/O/compute pipeline. After the authoritative router
    # has produced the complete expert union, fetch batch N+1 on one worker
    # while Metal consumes batch N. This predicts no routes and never changes
    # their order; the only extra residency is one governor-admitted batch.
    expert_batch_prefetch: bool = False
    # Number of authoritative future storage batches queued on the single
    # expert-I/O worker. One is the proven baseline; a larger explicit depth
    # can hide storage exposed by faster/coalesced expert compute while keeping
    # routes, fetch order, and arithmetic order unchanged.
    expert_batch_prefetch_depth: int = 1
    # Explicit storage-worker count for the ordered future queue. Consumption
    # remains authoritative-order even when two independent reads overlap.
    expert_batch_prefetch_workers: int = 1
    # Explicit measurement-only route analysis. Reconstruct adjacent-position
    # expert sets from the authoritative router output to quantify cache reuse
    # and speculative multi-position union growth. Disabled by default because
    # walking every route in Python is material at very large contexts.
    expert_route_overlap_telemetry: bool = False

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
            execution_profile=run.get("execution_profile", ""),
            native_ct_mxfp4=run.get("native_ct_mxfp4", False),
            kimi_k3_scale_sidecar_dir=run.get(
                "kimi_k3_scale_sidecar_dir", ""
            ),
            bf16_nf12_sidecar_dir=run.get(
                "bf16_nf12_sidecar_dir", ""
            ),
            bf16_nf12_uncached_reads=run.get(
                "bf16_nf12_uncached_reads", False
            ),
            bf16_nf12_direct_linear=run.get(
                "bf16_nf12_direct_linear", False
            ),
            safetensors_offset_order=run.get(
                "safetensors_offset_order", False
            ),
            min_weight_cache_mb=mem.get("min_weight_cache_mb", 1500),
            pin_embeddings=pinned.get("embeddings", True),
            pin_lm_head=pinned.get("lm_head", False),
            pin_first_layers=pinned.get("first_layers", 0),
            pin_last_layers=pinned.get("last_layers", 0),
            prefetch_depth=raw.get("prefetch", {}).get("depth", 0),
            prefetch_workers=raw.get("prefetch", {}).get("workers", 0),
            max_kv_mb=mem.get("max_kv_mb", 0),
            paged_kv_persist=run.get("paged_kv_persist", False),
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
            rerank_lm_head_source=run.get("rerank_lm_head_source", ""),
            rerank_lm_head_source_fingerprint=run.get(
                "rerank_lm_head_source_fingerprint", ""),
            rerank_lm_head_recall_probe_every=run.get(
                "rerank_lm_head_recall_probe_every", 0),
            rerank_lm_head_rank_capture_path=run.get(
                "rerank_lm_head_rank_capture_path", ""),
            rerank_lm_head_rank_capture_max_positions=run.get(
                "rerank_lm_head_rank_capture_max_positions", 1200),
            rerank_lm_head_rank_capture_max_per_request=run.get(
                "rerank_lm_head_rank_capture_max_per_request", 128),
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
            parallel_storage_reads=run.get(
                "parallel_storage_reads", mem.get(
                    "parallel_storage_reads", False)),
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
            qwen_mtp_speculative=run.get("qwen_mtp_speculative", False),
            hot_prompt_kv_persist_dir=run.get("hot_prompt_kv_persist_dir", ""),
            hot_prompt_kv_persist_max_checkpoints=run.get(
                "hot_prompt_kv_persist_max_checkpoints", 64),
            hot_prompt_kv_persist_max_mb=run.get(
                "hot_prompt_kv_persist_max_mb", 0),
            qwen_yarn_factor=run.get("qwen_yarn_factor", 0.0),
            prefill_chunk_size=run.get("prefill_chunk_size", 0),
            qwen35_prefill_chunk_ceiling=run.get(
                "qwen35_prefill_chunk_ceiling", 0),
            qwen35_serial_verify_exact_page_admission=run.get(
                "qwen35_serial_verify_exact_page_admission", False),
            qwen35_serial_verify_batched_mlp=run.get(
                "qwen35_serial_verify_batched_mlp", False),
            qwen35_serial_verify_suspend_lm_head=run.get(
                "qwen35_serial_verify_suspend_lm_head", False),
            qwen35_serial_verify_suspend_lm_head_min_prompt_tokens=run.get(
                "qwen35_serial_verify_suspend_lm_head_min_prompt_tokens",
                QWEN35_PHASE_HEAD_MIN_PROMPT_TOKENS),
            qwen_mixed_depth_endpoint_persist=run.get(
                "qwen_mixed_depth_endpoint_persist", False),
            prefill_last_token_separate=run.get(
                "prefill_last_token_separate", False),
            prefill_checkpoint_every=run.get("prefill_checkpoint_every", 0),
            adaptive_chunk_size=run.get("adaptive_chunk_size", False),
            adaptive_chunk_safe_bytes=run.get("adaptive_chunk_safe_bytes", 0),
            adaptive_chunk_escalate_growth_cap=run.get(
                "adaptive_chunk_escalate_growth_cap", False),
            embed_rows=run.get("embed_rows", False),
            stream_lm_head=run.get("stream_lm_head", False),
            governor=run.get("governor", True),
            vision_max_patches=run.get("vision_max_patches", 0),
            warm_start=run.get("warm_start", 0),
            mla_compressed_kv=run.get("mla_compressed_kv", True),
            mla_absorbed_decode=run.get("mla_absorbed_decode", False),
            glm53_sparse_absorbed_mla=run.get(
                "glm53_sparse_absorbed_mla", False),
            glm53_sparse_fused_attention=run.get(
                "glm53_sparse_fused_attention", False),
            glm53_sparse_fused_kv_int8=run.get(
                "glm53_sparse_fused_kv_int8", False),
            glm53_coalesced_expert_positions=run.get(
                "glm53_coalesced_expert_positions", False),
            glm53_coalesced_expert_max_positions=run.get(
                "glm53_coalesced_expert_max_positions", 512),
            glm53_incremental_dsa_pool=run.get(
                "glm53_incremental_dsa_pool", False),
            glm53_compiled_kda_prefill=run.get(
                "glm53_compiled_kda_prefill", False),
            glm53_compiled_kda_segment=run.get(
                "glm53_compiled_kda_segment", 32),
            glm53_native_fused_kda_prefill=run.get(
                "glm53_native_fused_kda_prefill", False),
            glm53_layer_stationary_host_spool=run.get(
                "glm53_layer_stationary_host_spool", False),
            glm_dsa_key_tile_size=run.get(
                "glm_dsa_key_tile_size", 256),
            glm_dsa_index_step_size=run.get(
                "glm_dsa_index_step_size", 0),
            glm_dsa_index_preallocate=run.get(
                "glm_dsa_index_preallocate", False),
            glm_dsa_selection_query_tile_size=run.get(
                "glm_dsa_selection_query_tile_size", 0),
            glm_dsa_dense_mlp_tile_size=run.get(
                "glm_dsa_dense_mlp_tile_size", 0),
            glm_dsa_sparse_absorbed_mla=run.get(
                "glm_dsa_sparse_absorbed_mla", False),
            glm_dsa_mla_kv_spill_dir=run.get(
                "glm_dsa_mla_kv_spill_dir", ""),
            kimi_k3_compressed_mla=run.get(
                "kimi_k3_compressed_mla", False),
            kimi_k3_absorbed_mla=run.get(
                "kimi_k3_absorbed_mla", False),
            kimi_k3_mla_key_tile_size=run.get(
                "kimi_k3_mla_key_tile_size", 2048),
            kimi_k3_fused_attnres_tile_size=run.get(
                "kimi_k3_fused_attnres_tile_size", 0),
            kimi_k3_attnres_spill_dir=run.get(
                "kimi_k3_attnres_spill_dir", ""),
            kimi_k3_kda_spill_dir=run.get(
                "kimi_k3_kda_spill_dir", ""),
            kimi_k3_mla_kv_spill_dir=run.get(
                "kimi_k3_mla_kv_spill_dir", ""),
            kimi_k3_dense_mlp_tile_size=run.get(
                "kimi_k3_dense_mlp_tile_size", 0),
            kimi_k3_prefill_tile_policy=run.get(
                "kimi_k3_prefill_tile_policy", "fixed"),
            kimi_k3_prefill_long_context_tokens=run.get(
                "kimi_k3_prefill_long_context_tokens", 256),
            kimi_k3_prefill_short_tile_width=run.get(
                "kimi_k3_prefill_short_tile_width", 256),
            kimi_k3_dense_mlp_short_tile_size=run.get(
                "kimi_k3_dense_mlp_short_tile_size", 0),
            warm_mb=run.get("warm_mb", 0),
            final_dead_token_elim=run.get("final_dead_token_elim", True),
            router_lookahead=run.get("router_lookahead", False),
            expert_predictive_prefetch=run.get(
                "expert_predictive_prefetch", False),
            expert_prefetch_idle_only=run.get(
                "expert_prefetch_idle_only", True),
            context_bound=run.get("context_bound", 0),
            expert_fetch_batch=run.get("expert_fetch_batch", 0),
            expert_compute_batch=run.get("expert_compute_batch", 0),
            decode_expert_fetch_batch=run.get("decode_expert_fetch_batch", 0),
            expert_batch_prefetch=run.get("expert_batch_prefetch", False),
            qwen4_expert_batch_prefetch_prefill_only=run.get(
                "qwen4_expert_batch_prefetch_prefill_only", False),
            expert_batch_prefetch_depth=run.get(
                "expert_batch_prefetch_depth", 1),
            expert_batch_prefetch_workers=run.get(
                "expert_batch_prefetch_workers", 1),
            expert_route_overlap_telemetry=run.get(
                "expert_route_overlap_telemetry", False
            ),
        )


def _apply_runtime_expert_top_k(rc: RuntimeConfig, cfg) -> None:
    """Validate and copy an explicitly lossy runtime routing schedule."""
    raw_schedule = rc.expert_top_k_by_layer
    if not isinstance(raw_schedule, (list, tuple)):
        raise ValueError(
            "expert_top_k_by_layer must be a list or tuple of integers")
    supported = cfg.model_type in ("olmoe", "qwen3_5_moe", "kimi_k3")
    if raw_schedule and not supported:
        raise ValueError(
            "expert_top_k_by_layer is supported only for OLMoE, "
            "Qwen3.5/3.6 MoE, and Kimi K3 checkpoints")
    schedule = (
        validate_expert_top_k_by_layer(cfg, raw_schedule)
        if supported else ()
    )
    rc.expert_top_k_by_layer = schedule
    if supported:
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
    # F95 (2026-07-21): the prefill_chunk_size this slot's KV/recurrent state
    # was actually built with. A continuation matching this slot MUST reuse
    # this exact value (hot_prompt_kv's fixed-chunk-per-lineage invariant),
    # not whatever the engine's current default happens to be -- that's what
    # makes per-CONVERSATION chunk-size adaptivity safe: a fresh conversation
    # (no matching slot) samples live memory and can pick a bigger chunk when
    # healthy, while a continuing one stays pinned to whatever built it,
    # never silently drifting mid-lineage. Required (no default) so every
    # construction site must decide this explicitly rather than risk a
    # stale/wrong value slipping through unnoticed.
    chunk_size: int
    exact_hidden: mx.array | None = None
    # K3's prompt-length policy also changes the dense-MLP tile. A content-
    # blind bucket id prevents an endpoint built by the short schedule from
    # being reused after a continuation crosses into the long schedule (or
    # vice versa). Empty preserves all pre-adaptive/non-K3 slots.
    kimi_k3_prefill_schedule: str = ""
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
    # True only for slots reconstructed from the durable journal during
    # engine startup.  They are resident by the time lookup runs, so the
    # ordinary source remains ``memory``; this provenance bit lets production
    # telemetry prove that the first post-restart hit really crossed disk.
    persisted_preload: bool = False


# Trunk weights MLX's fused MXFP8 kernel can read directly, so the dequant
# never happens. wq_b, wo_b and wo_a are 60% of a trunk layer between them;
# wo_a reaches the kernel as n_groups contiguous row blocks rather than one
# operand, since its grouped einsum is really that many independent matmuls.
_DSV4_FUSED_FP8_SUFFIXES = (
    "attn.wq_a.weight",
    "attn.wq_b.weight",
    "attn.wkv.weight",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.weight",
)


class StreamingEngine:
    def __init__(self, model_dir: str | Path, rc: RuntimeConfig | None = None):
        self.rc = rc or RuntimeConfig()
        self.rc.execution_profile = str(
            self.rc.execution_profile or "").strip().lower()
        self.rc.kimi_k3_prefill_tile_policy = str(
            self.rc.kimi_k3_prefill_tile_policy or "fixed").strip().lower()
        # These are the configured long-context values. The active RuntimeConfig
        # fields are allowed to change per request, so retain immutable copies
        # before the first adaptive selection or memory retry mutates them.
        self._k3_prefill_long_tile_width = int(self.rc.prefill_chunk_size)
        self._k3_dense_mlp_long_tile_size = int(
            self.rc.kimi_k3_dense_mlp_tile_size)
        if self.rc.execution_profile not in telemetry.RequestProfiler.LEVELS:
            raise ValueError(
                "execution_profile must be '', 'layers', or 'ops'")
        if self.rc.mlx_cache_limit_mb <= 0:
            raise ValueError("mlx_cache_limit_mb must be positive")
        # MLX's buffer cache is NOT counted in our weight budget and can balloon
        # by gigabytes under paging churn (measured 2.3 GB), pushing the machine
        # over the macOS wired-memory line. Exact paged-KV profiles use a much
        # smaller server-configured cap than the ordinary 1-GiB default.
        mx.set_cache_limit(self.rc.mlx_cache_limit_mb * 1_000_000)
        if self.rc.stepped_kv_threshold < 0:
            raise ValueError("stepped_kv_threshold must be >= 0")
        if self.rc.kimi_k3_mla_key_tile_size < 0:
            raise ValueError("kimi_k3_mla_key_tile_size must be >= 0")
        if self.rc.glm_dsa_key_tile_size <= 0:
            raise ValueError("glm_dsa_key_tile_size must be positive")
        if self.rc.glm_dsa_index_step_size < 0:
            raise ValueError("glm_dsa_index_step_size must be non-negative")
        if (self.rc.glm_dsa_index_preallocate
                and self.rc.glm_dsa_index_step_size <= 0):
            raise ValueError(
                "glm_dsa_index_preallocate requires a positive index step")
        if self.rc.glm_dsa_selection_query_tile_size < 0:
            raise ValueError(
                "glm_dsa_selection_query_tile_size must be non-negative")
        if self.rc.glm_dsa_dense_mlp_tile_size < 0:
            raise ValueError(
                "glm_dsa_dense_mlp_tile_size must be non-negative")
        if (
            self.rc.glm_dsa_mla_kv_spill_dir
            and not self.rc.mla_compressed_kv
        ):
            raise ValueError(
                "glm_dsa_mla_kv_spill_dir requires compressed MLA"
            )
        if self.rc.glm53_compiled_kda_segment not in (16, 32, 64, 128):
            raise ValueError(
                "glm53_compiled_kda_segment must be 16, 32, 64, or 128")
        if (
            self.rc.glm_dsa_sparse_absorbed_mla
            and not self.rc.glm_dsa_mla_kv_spill_dir
        ):
            raise ValueError(
                "glm_dsa_sparse_absorbed_mla requires the explicit long-context spill path"
            )
        if self.rc.kimi_k3_fused_attnres_tile_size < 0:
            raise ValueError(
                "kimi_k3_fused_attnres_tile_size must be >= 0"
            )
        if (
            self.rc.kimi_k3_attnres_spill_dir
            and not self.rc.kimi_k3_fused_attnres_tile_size
        ):
            raise ValueError(
                "kimi_k3_attnres_spill_dir requires fused AttnRes tiling"
            )
        if (
            self.rc.kimi_k3_mla_kv_spill_dir
            and not self.rc.kimi_k3_compressed_mla
        ):
            raise ValueError(
                "kimi_k3_mla_kv_spill_dir requires compressed MLA"
            )
        if self.rc.kimi_k3_dense_mlp_tile_size < 0:
            raise ValueError(
                "kimi_k3_dense_mlp_tile_size must be >= 0"
            )
        if (
            self.rc.kimi_k3_compiled_kda_prefill
            and self.rc.kimi_k3_native_fused_kda_prefill
        ):
            raise ValueError(
                "compiled and native-fused K3 KDA prefill are mutually "
                "exclusive"
            )
        if sum(map(bool, (
            self.rc.qwen_compiled_delta_prefill,
            self.rc.qwen_chunked_delta_prefill,
            self.rc.qwen_native_fused_delta_prefill,
        ))) > 1:
            raise ValueError(
                "compiled, chunked, and native-fused Qwen DeltaNet prefill "
                "are mutually exclusive"
            )
        if self.rc.kimi_k3_prefill_tile_policy not in (
            "fixed", "prompt-length"
        ):
            raise ValueError(
                "kimi_k3_prefill_tile_policy must be 'fixed' or "
                "'prompt-length'"
            )
        if self.rc.kimi_k3_prefill_long_context_tokens <= 0:
            raise ValueError(
                "kimi_k3_prefill_long_context_tokens must be positive"
            )
        if not 1 <= self.rc.kimi_k3_prefill_short_tile_width <= 4096:
            raise ValueError(
                "kimi_k3_prefill_short_tile_width must be in [1, 4096]"
            )
        if not 0 <= self.rc.kimi_k3_dense_mlp_short_tile_size <= 4096:
            raise ValueError(
                "kimi_k3_dense_mlp_short_tile_size must be in [0, 4096]"
            )
        if (
            self.rc.kimi_k3_prefill_tile_policy == "prompt-length"
            and not 1 <= self._k3_prefill_long_tile_width <= 4096
        ):
            raise ValueError(
                "prompt-length K3 prefill requires the configured long tile "
                "width to be in [1, 4096]"
            )
        if (
            self.rc.kimi_k3_prefill_tile_policy == "prompt-length"
            and (self.rc.prompt_kv_dir or self.rc.hot_prompt_kv_persist_dir)
        ):
            raise ValueError(
                "prompt-length K3 prefill currently supports only cold or "
                "in-memory hot KV; durable prompt-KV stores require a fixed "
                "schedule"
            )
        if (
            self.rc.kimi_k3_absorbed_mla
            and not self.rc.kimi_k3_compressed_mla
        ):
            raise ValueError(
                "kimi_k3_absorbed_mla requires "
                "kimi_k3_compressed_mla"
            )
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
        if self.rc.rerank_lm_head_recall_probe_every < 0:
            raise ValueError(
                "rerank_lm_head_recall_probe_every must be non-negative")
        if self.rc.rerank_lm_head_rank_capture_path:
            from .lm_head_recall_capture import (
                CAPTURE_MAX_PER_REQUEST_LIMIT,
                CAPTURE_MAX_POSITIONS_LIMIT,
                PROMOTION_K,
                PROMOTION_MIN_POSITIONS,
            )

            if not self.rc.rerank_lm_head:
                raise ValueError(
                    "LM-head rank capture requires candidate reranking")
            if self.rc.rerank_lm_head_candidates != PROMOTION_K:
                raise ValueError(
                    f"LM-head promotion capture requires K={PROMOTION_K}")
            if not self.rc.rerank_lm_head_source_fingerprint:
                raise ValueError(
                    "LM-head rank capture requires a fingerprinted exact source")
            if not PROMOTION_MIN_POSITIONS <= (
                    self.rc.rerank_lm_head_rank_capture_max_positions
                    ) <= CAPTURE_MAX_POSITIONS_LIMIT:
                raise ValueError(
                    "LM-head rank capture max positions must be in "
                    f"[{PROMOTION_MIN_POSITIONS}, "
                    f"{CAPTURE_MAX_POSITIONS_LIMIT}]")
            if not 1 <= (
                    self.rc.rerank_lm_head_rank_capture_max_per_request
                    ) <= CAPTURE_MAX_PER_REQUEST_LIMIT:
                raise ValueError(
                    "LM-head rank capture per-request positions must be in "
                    f"[1, {CAPTURE_MAX_PER_REQUEST_LIMIT}]")
        if self.rc.rerank_lm_head_source or \
                self.rc.rerank_lm_head_source_fingerprint:
            fingerprint = self.rc.rerank_lm_head_source_fingerprint
            if (not self.rc.rerank_lm_head_source
                    or len(fingerprint) != 64
                    or any(character not in "0123456789abcdef"
                           for character in fingerprint)):
                raise ValueError(
                    "row-paged reranked LM head requires an exact source and "
                    "64-character lowercase fingerprint")
        if self.rc.adaptive_chunk_safe_bytes < 0:
            raise ValueError("adaptive_chunk_safe_bytes must be >= 0")
        if self.rc.qwen35_prefill_chunk_ceiling not in (
                QWEN35_PREFILL_CHUNK_CEILINGS):
            raise ValueError(
                "qwen35_prefill_chunk_ceiling must be one of "
                "0, 1, 8, 32, 128, or 512")
        if self.rc.prompt_kv_min_tokens < 0:
            raise ValueError("prompt_kv_min_tokens must be >= 0")
        if self.rc.prompt_kv_journal_chunk_size <= 0:
            raise ValueError("prompt_kv_journal_chunk_size must be positive")
        if self.rc.hot_prompt_kv_persist_max_mb < 0:
            raise ValueError("hot_prompt_kv_persist_max_mb must be non-negative")
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
        if self.rc.paged_kv_persist and (
                not self.rc.max_kv_mb
                or not self.rc.hot_prompt_kv
                or not self.rc.hot_prompt_kv_persist_dir
                or not self.rc.release_paged_kv_after_generate):
            raise ValueError(
                "paged_kv_persist requires paged KV, hot_prompt_kv, a durable "
                "directory, and post-generation release")
        if (self.rc.paged_kv_persist
                and self.rc.qwen_fused_boundary_scaffold_prefill):
            raise ValueError(
                "paged durable KV requires an explicit stable-boundary sweep")
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
        if self.rc.qwen_postgen_min_available_mb < 0:
            raise ValueError(
                "qwen_postgen_min_available_mb must be non-negative")
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
            parallel_storage_reads=self.rc.parallel_storage_reads,
            native_ct_mxfp4=self.rc.native_ct_mxfp4,
            kimi_k3_scale_sidecar_dir=self.rc.kimi_k3_scale_sidecar_dir,
            bf16_nf12_sidecar_dir=self.rc.bf16_nf12_sidecar_dir,
            bf16_nf12_uncached_reads=(
                self.rc.bf16_nf12_uncached_reads
            ),
            bf16_nf12_direct_linear=(
                self.rc.bf16_nf12_direct_linear
            ),
            safetensors_offset_order=(
                self.rc.safetensors_offset_order
            ),
        )
        # WeightStore may have re-resolved a stale SMB mount from Plex to
        # Plex-N.  Every later checkpoint-relative path must follow that same
        # healthy directory; mixing the recovered weights/config with a stale
        # tokenizer, sidecar, fingerprint, or predictor path defeats F24.
        self._model_dir = self.store.dir
        self.cfg = self.store.config
        _apply_runtime_expert_top_k(self.rc, self.cfg)
        self._qwen4_ple_rows = None
        if self.cfg.model_type == "qwen4_exp":
            from .qwen4_exp_ple_rows import Qwen4ExpPLERowStore

            # The source witness binds all 33 PLE-bearing release shards to the
            # pinned Hub revision. Row paging is mandatory: loading the table
            # as an ordinary layer weight would require roughly 95 GiB.
            self._qwen4_ple_rows = Qwen4ExpPLERowStore(
                self._model_dir, row_cache=8192, require_release_hash=True,
                read_workers=self.rc.qwen4_ple_read_workers)
        if (self.rc.paged_kv_persist
                and self.cfg.model_type not in ("qwen3_5", "qwen3_5_moe")):
            raise ValueError(
                "paged_kv_persist requires a qwen3_5/qwen3_5_moe hybrid "
                f"checkpoint, got {self.cfg.model_type!r}")
        if (self.cfg.model_type in (
                "kimi_linear", "kimi_k3", "qwen3_5_moe", "qwen3_5",
                "qwen4_exp", "glm5_next")
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
        if self.cfg.model_type in (
                "glm_moe_dsa", "kimi_k25", "glm4_moe_lite", "glm5_next"):
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
                "glm_moe_dsa", "kimi_linear", "kimi_k3", "kimi_k25",
                "qwen3_5_moe", "qwen4_exp", "glm5_next")
                and self.rc.expert_fetch_batch <= 0):
            # F74-v2 is a safety default for every construction path, including
            # direct experiments and YAML. Leaving zero as "unbounded" silently
            # bypassed the server's GLM-specific protection and recreated the
            # 16-22 GB union lifetime. q=1 remains the fail-closed default for
            # dense or expand-on-load representations. Other architectures
            # retain zero semantics. F92: Kimi Linear/K2.5 have
            # 256/384 experts each, the same "prefill floods the union" risk
            # GLM was fixed for here (measured 2026-07-18: an unbounded fetch
            # on a 15-token prompt requested ~2.8GB in one shot and was
            # correctly refused by the governor). Qwen3.6 likewise has 256
            # experts per layer and can route a near-complete union during a
            # multi-position prefill, so the same lifetime bound applies.
            #
            # F134: native compressed-tensors MXFP4 is different: the compact
            # page is the released representation that survives in cache, not
            # a temporary BF16 expansion. Derive a bounded coalescing batch
            # from those physical resident bytes instead of model identity or
            # one prompt's routed union. Real K3 five-position, first-four-
            # layer A/B: q=8 -> q=16 also won on two unrelated prompts with
            # identical hidden bytes and the same 5.650GB Metal peak. Full
            # 93-layer q=1/q=8/q=16 gates measured 317.299/265.037/260.392s
            # with the same token, bytes, and 6.744GB peak. Native MXFP4 itself
            # remains explicit opt-in, so this does not turn a narrow result
            # into a new automatic public-server path.
            self._auto_compact_expert_batch = 0
            if (self.store.native_ct_mxfp4
                    or getattr(self.store, "glm53_fp8_direct_qmv", False)
                    or getattr(self.store, "qwen4_fp8_direct_qmv", False)):
                inter = (
                    getattr(self.cfg, "moe_intermediate_size", None)
                    or self.cfg.intermediate_size)
                expert_hidden = (
                    self.cfg.moe_latent_hidden_size or self.cfg.hidden_size)
                compact_page_bytes = int(
                    3 * expert_hidden * inter
                    * self.store.expert_resident_bytes_per_weight)
                self.rc.expert_fetch_batch = compact_expert_io_batch_size(
                    compact_page_bytes,
                    self.rc.max_weight_cache_mb * 1_000_000,
                )
                self._auto_compact_expert_batch = (
                    self.rc.expert_fetch_batch)
            else:
                self.rc.expert_fetch_batch = 1
        else:
            self._auto_compact_expert_batch = 0
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
            if self.store.mtplx_mtp_sidecar:
                # The MTPLX body is already standard MLX MXFP4, while its
                # separately indexed native-MTP block is intentionally the
                # released BF16 draft head.  Keep that sidecar in its declared
                # format instead of applying the body's runtime quant policy a
                # second time; accepted tokens remain target-verified either
                # way, but this preserves the artifact's measured draft
                # contract and expected acceptance rate.
                def transform(name, value):
                    if name.startswith("mtp."):
                        return value
                    return quant_policy.transform(name, value)
        warm = None
        if self.rc.warm_mb:
            from .warm_tier import WarmTier

            warm = WarmTier(self.rc.warm_mb * 1_000_000)
        self.cache = WeightCache(self.store, self.rc.max_weight_cache_mb * 1_000_000, transform, warm,
                                  max_fetch_batch=self.rc.expert_fetch_batch)
        self.timer = telemetry.Timer()
        # Created afresh by generate(); never shared across requests. Keeping
        # the disabled state as None makes ordinary inference pay only a single
        # predictable branch at existing layer boundaries.
        self._request_profiler: telemetry.RequestProfiler | None = None
        # F42: per-expert page byte estimate for pre-allocation reservations.
        # moe_intermediate_size when the config has it, else the dense size
        # (over-estimate = conservative); MXFP4 stores ~0.53 B/weight.
        inter = getattr(self.cfg, "moe_intermediate_size", None) or self.cfg.intermediate_size
        # F128: Kimi K3's real Stable LatentMoE runs routed experts on a
        # compressed latent width (config.routed_expert_hidden_size, 3584
        # for the real checkpoint), not the full hidden_size (7168) --
        # confirmed against real downloaded expert tensor shapes. Using
        # hidden_size here doubled this estimate for K3 specifically (every
        # other model this project supports leaves moe_latent_hidden_size
        # at its 0 default, so this is unchanged for them).
        expert_hidden = self.cfg.moe_latent_hidden_size or self.cfg.hidden_size
        if getattr(self.store, "dsv4_native_mxfp4", False):
            # A DeepSeek V4 routed expert kept in its released MXFP4 form is
            # 4-bit codes plus one E8M0 byte per group of 32, not bf16. The
            # bf16 figure below is 50.33MB against an actual 13.4MB, and that
            # 4x lands twice: the governor reserves it on every expert fetch,
            # and the trunk pin planner subtracts a whole batch of it from the
            # pin budget, costing pinnable layers for memory nothing occupies.
            resident_bytes_per_weight = 0.5 + 1.0 / 32.0
        elif self.store.on_disk_quantized:
            resident_bytes_per_weight = self.store.quantized_bytes_per_weight
        elif (self.store.native_ct_mxfp4
              or getattr(self.store, "glm53_fp8_direct_qmv", False)
              or getattr(self.store, "qwen4_fp8_direct_qmv", False)):
            resident_bytes_per_weight = (
                self.store.expert_resident_bytes_per_weight)
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
            3 * expert_hidden * inter * 2)
        self._expert_page_bytes = int(
            3 * expert_hidden * inter * resident_bytes_per_weight)
        self._expert_storage_page_bytes = int(
            3 * expert_hidden * inter
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
        import os as _os

        from collections import OrderedDict as _OrderedDict

        self._dsv4_snapshots: dict = _OrderedDict()
        self._dsv4_prefix_checkpoints: dict = _OrderedDict()
        self._dsv4_prefix_slots = int(
            _os.environ.get("VMODEL_DSV4_PREFIX_SLOTS", "2"))
        self._dsv4_checkpoint = None
        self._dsv4_indexer = (
            _os.environ.get("VMODEL_DSV4_INDEXER") == "1")
        _carrier = _os.environ.get("VMODEL_DSV4_CARRIER_DTYPE", "")
        self._dsv4_carrier_dtype = (
            mx.bfloat16 if _carrier == "bfloat16"
            else mx.float16 if _carrier == "float16" else None)
        self._dsv4_snapshot_slots = int(
            _os.environ.get("VMODEL_DSV4_PROMPT_SLOTS", "4"))
        self._dsv4_prompt_reuse = (
            _os.environ.get("VMODEL_DSV4_PROMPT_REUSE") == "1")
        self._dspark_capture = None
        self._dspark_targets = frozenset(
            int(v) for v in (self.cfg.dspark_target_layer_ids or ()))
        self._dsv4_fused_fp8 = (
            _os.environ.get("VMODEL_DSV4_FUSED_FP8") == "1"
            and _os.environ.get("VMODEL_DSV4_PACKED_TRUNK") == "1")
        self._dsv4_packed_trunk = (
            _os.environ.get("VMODEL_DSV4_PACKED_TRUNK") == "1")
        self._dsv4_expert_retain = (
            _os.environ.get("VMODEL_DSV4_EXPERT_RETAIN") == "1"
            and _os.environ.get("VMODEL_DSV4_NATIVE_MXFP4") == "1")
        self._expert_fetch_page_bytes = (
            self._expert_page_bytes
            if (self.store.on_disk_quantized
                or self.store.native_ct_mxfp4
                or getattr(self.store, "glm53_fp8_direct_qmv", False)
                or getattr(self.store, "qwen4_fp8_direct_qmv", False)
                or not self.rc.quant_bits)
            else dense_expert_page_bytes + self._expert_page_bytes
        )
        self._layer_transient = 0  # F42: measured compute-scratch high-water mark
        self._prefill_layer_transient = 0
        self._prefill_layer_transient_by_positions: dict[int, int] = {}
        self._decode_layer_transient = 0
        # F132: heterogeneous stacks can have radically different compute
        # scratch requirements.  Kimi K3's first dense MLP, for example, has
        # no relationship to the packed routed-expert path used by every
        # following MoE layer.  Keep the aggregate high-water marks above for
        # request/KV admission, but use this operation-signature map for the
        # imminent per-layer reservation.
        self._layer_transient_by_signature: dict[tuple[int, str], int] = {}
        # The first execution of an operation signature includes one-time
        # Metal/JIT setup. Keep it as the bootstrap reserve for the second
        # occurrence, then use the maximum of recurring observations so a
        # compile-only spike cannot poison all remaining layers.
        self._layer_transient_observation_counts: dict[
            tuple[int, str], int
        ] = {}
        self._layer_transient_recurring_max: dict[
            tuple[int, str], int
        ] = {}
        # Serial speculative verification executes one-position arithmetic but
        # retains several position outputs/endpoints. Its high-water must not
        # train the ordinary one-token decode reserve, or the following token
        # over-evicts useful weights.
        self._serial_verify_layer_transient: dict[
            tuple[int, str], int
        ] = {}
        self._serial_verify_layer_transient_counts: dict[
            tuple[int, str], int
        ] = {}
        self._serial_verify_layer_transient_recurring_max: dict[
            tuple[int, str], int
        ] = {}
        self._qwen35_serial_verify_batched_mlp_layers = 0
        self._qwen35_serial_verify_batched_mlp_positions = 0
        self._qwen35_serial_verify_batched_mlp_s = 0.0
        self._qwen35_serial_verify_page_prepare_s = 0.0
        self._qwen35_serial_verify_cache_prepare_s = 0.0
        self._qwen35_serial_verify_page_reserve_s = 0.0
        self._qwen35_serial_verify_reserve_s = 0.0
        self._qwen35_serial_verify_weight_wait_s = 0.0
        self._qwen35_serial_verify_linear_layer_compute_s = 0.0
        self._qwen35_serial_verify_full_layer_compute_s = 0.0
        self._qwen35_serial_verify_head_s = 0.0
        self._qwen35_serial_verify_head_suspend_calls = 0
        self._qwen35_serial_verify_head_suspend_bytes = 0
        self._qwen35_serial_verify_head_suspend_active_released_bytes = 0
        self._qwen35_serial_verify_head_suspend_active_peak_bytes = 0
        self._qwen35_serial_verify_head_suspend_s = 0.0
        self._qwen35_serial_verify_head_restore_calls = 0
        self._qwen35_serial_verify_head_restore_successes = 0
        self._qwen35_serial_verify_head_restore_refusals = 0
        self._qwen35_serial_verify_head_restore_s = 0.0
        self._qwen35_lm_head_pin_suspended = False
        self._qwen35_lm_head_suspend_request_active = False
        self._qwen4_lm_head_pin_suspended = False
        self._qwen4_phase_head_bytes = 0
        self._qwen4_phase_head_suspend_calls = 0
        self._qwen4_phase_head_suspend_bytes = 0
        self._qwen4_phase_head_suspend_s = 0.0
        self._qwen4_phase_head_restore_calls = 0
        self._qwen4_phase_head_restore_successes = 0
        self._qwen4_phase_head_restore_refusals = 0
        self._qwen4_phase_head_restore_s = 0.0
        self._qwen4_serial_verify_head_suspend_calls = 0
        self._qwen4_serial_verify_head_suspend_bytes = 0
        self._qwen4_serial_verify_head_restore_trim_bytes = 0
        self._qwen4_serial_verify_exact_bf16_calls = 0
        self._qwen4_serial_verify_exact_bf16_rows = 0
        self._qwen4_serial_verify_exact_bf16_fallback_calls = 0
        self._qwen4_serial_verify_exact_bf16_fallback_reasons = {}
        self._layer_transient_margin = 400_000_000
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
        # A DSpark bootstrap can subscribe to the ordinary prefill schedule so
        # target feature taps are consumed while the streamed/layer-stationary
        # sweep is already resident. This avoids a second complete target
        # prefill. Explicit per-call ``tap_layers`` (verification) remains
        # independent and takes precedence.
        self._dspark_tap_collector = None
        # F189: optional exact recurrent endpoints retained by the most recent
        # layer-major verify sweep. They are consumed immediately after target
        # acceptance is known; ordinary calls leave this empty.
        self._serial_kda_endpoints = None
        self._serial_kda_endpoint_retained_bytes = 0
        self._serial_qwen4_endpoints = None
        self._serial_qwen4_endpoint_retained_bytes = 0
        self._serial_kda_factors = None
        self._serial_kda_factor_retained_bytes = 0
        self._dspark_expert_prefetch_plan = None
        self._dspark_expert_prefetch_depth = 0
        # F43: a declared context bound <= index_topk means the DSA indexer can
        # never deselect anything — elide its weights and state entirely.
        self._dsa_elided = bool(
            self.cfg.model_type in ("glm_moe_dsa", "glm5_next")
            and self.cfg.index_topk
            and self.rc.context_bound and self.rc.context_bound <= self.cfg.index_topk
        )

        # ---- pin persistent tensors ----
        # Embeddings and final norm are touched every token; norm is bytes-sized so
        # it is pinned unconditionally alongside them.
        self._embed_rows = None
        if (self.rc.embed_rows and not self.cfg.tie_word_embeddings
                and not self.store.is_quantized("model.embed_tokens.weight")
                and self.store.gguf is None):
            from .embed_rows import EmbedRows

            self._embed_rows = EmbedRows(self._model_dir, self.store, self.cfg.hidden_size)

        self._streamed_lm_head = None
        if (self.rc.stream_lm_head and not self.cfg.tie_word_embeddings
                and not self.store.is_quantized("lm_head.weight")
                and self.store.has("lm_head.weight")
                and not self.store.vpack2 and not self.store.packed
                and self.store.gguf is None):
            from .lm_head_stream import StreamedLMHead

            self._streamed_lm_head = StreamedLMHead(
                self.store.dir, self.store.weight_map,
                real_name=self.store._real_name.get("lm_head.weight", "lm_head.weight"))

        phase_scoped_qwen35_head = bool(
            self.rc.qwen35_serial_verify_suspend_lm_head)
        phase_scoped_qwen4_head = bool(self.rc.qwen4_phase_lm_head)
        if phase_scoped_qwen35_head and (
                self.cfg.model_type not in ("qwen3_5", "qwen3_5_moe")
                or not self.rc.pin_lm_head
                or self.rc.rerank_lm_head
                or self._streamed_lm_head is not None
                or self.cfg.tie_word_embeddings
                or not self.store.has("lm_head.weight")):
            raise ValueError(
                "qwen35_serial_verify_suspend_lm_head requires an untied, "
                "non-streamed, non-reranked pinned Qwen LM head")
        if phase_scoped_qwen4_head and (
                self.cfg.model_type != "qwen4_exp"
                or not self.rc.pin_lm_head
                or self.rc.rerank_lm_head
                or self._streamed_lm_head is not None
                or self.cfg.tie_word_embeddings
                or not self.store.has("lm_head.weight")):
            raise ValueError(
                "qwen4_phase_lm_head requires an untied, non-streamed, "
                "non-reranked pinned Qwen4 LM head")
        if (self.rc.qwen4_serial_verify_suspend_lm_head
                and not phase_scoped_qwen4_head):
            raise ValueError(
                "qwen4_serial_verify_suspend_lm_head requires "
                "qwen4_phase_lm_head")
        if phase_scoped_qwen35_head and phase_scoped_qwen4_head:
            raise ValueError("Qwen phase-scoped LM-head modes conflict")
        phase_scoped_qwen_head = bool(
            phase_scoped_qwen35_head or phase_scoped_qwen4_head)

        if self.cfg.model_type == "qwen4_exp":
            pin_names = self.store.names_with_prefix(
                "model.hyper_connection_mixer.")
            if not pin_names:
                raise ValueError("Qwen4-Exp final hyper mixer is missing")
        else:
            pin_names = ["model.norm.weight"]
        if self.rc.pin_embeddings and self._embed_rows is None:
            pin_names.append("model.embed_tokens.weight")
        if ((self.rc.pin_lm_head or self.rc.rerank_lm_head)
                and self._streamed_lm_head is None
                and not self.cfg.tie_word_embeddings
                and self.store.has("lm_head.weight")
                and not phase_scoped_qwen_head):
            pin_names.append("lm_head.weight")
        # F128: kimi_k3's AttnRes needs one final readout applied once after
        # ALL layers, before model.norm (real KimiLinearModel._apply_output_
        # attn_res) -- its two weights are tiny ((1,H) and (H,)) top-level
        # tensors, same pinning treatment as model.norm.weight itself.
        if self.cfg.model_type == "kimi_k3":
            pin_names.append("model.output_attn_res_proj.weight")
            pin_names.append("model.output_attn_res_norm.weight")
        # F213: DeepSeek V4's final hyper-connection reduction, applied once
        # after every layer and before model.norm. Small top-level tensors,
        # same pinning treatment as model.norm.weight.
        if self.cfg.model_type == "deepseek_v4":
            pin_names.extend(["model.hc_head_fn", "model.hc_head_scale",
                              "model.hc_head_base"])
        persistent = self.cache.pin("persistent", pin_names)
        if phase_scoped_qwen_head:
            if phase_scoped_qwen35_head:
                phase_head_bytes = self.store.mlx_quantized_resident_bytes(
                    ["lm_head.weight"])
            else:
                phase_head_bytes = self.store.storage_bytes(
                    ["lm_head.weight"])
            if phase_head_bytes <= 0:
                raise ValueError(
                    "phase-scoped Qwen LM head requires exactly sizeable "
                    "checkpoint metadata")
            if self.store.storage_bytes_unknown(["lm_head.weight"]):
                raise ValueError(
                    "phase-scoped Qwen LM head metadata is incomplete")
            if phase_scoped_qwen35_head:
                self.cache.register_suspended_pin(
                    "qwen35:lm_head:persistent", phase_head_bytes)
                self._qwen35_lm_head_pin_suspended = True
            else:
                self.cache.register_suspended_pin(
                    "qwen4:lm_head:persistent", phase_head_bytes,
                    allow_over_capacity=True)
                self._qwen4_lm_head_pin_suspended = True
                self._qwen4_phase_head_bytes = int(phase_head_bytes)

        self._embed_w = persistent.get("model.embed_tokens.weight")
        self._norm_w = persistent.get("model.norm.weight")
        self._lm_head_w = (
            None
            if phase_scoped_qwen_head
            else persistent.get("lm_head.weight")
        )
        self._hc_head_fn = persistent.get("model.hc_head_fn")
        self._hc_head_scale = persistent.get("model.hc_head_scale")
        self._hc_head_base = persistent.get("model.hc_head_base")
        self._qwen4_final_mixer_w = {
            name: value for name, value in persistent.items()
            if name.startswith("model.hyper_connection_mixer.")
        }
        self._output_attn_res_proj_w = persistent.get("model.output_attn_res_proj.weight")
        self._output_attn_res_norm_w = persistent.get("model.output_attn_res_norm.weight")
        self._reranked_lm_head_bytes = 0
        self._reranked_lm_head_exact_resident_bytes = 0
        self._reranked_lm_head_source_fingerprint = ""
        self._reranked_lm_head_approx_fingerprint = ""
        if self.rc.rerank_lm_head:
            from .quant import (
                QTensor, make_reranked_q_head,
                make_row_paged_reranked_q_head)

            if self.cfg.tie_word_embeddings:
                raise ValueError(
                    "rerank_lm_head currently requires an untied exact LM head")
            if self._streamed_lm_head is not None or self._lm_head_w is None:
                raise ValueError(
                    "rerank_lm_head requires an available target LM head")
            if self.rc.rerank_lm_head_source:
                if not isinstance(self._lm_head_w, QTensor):
                    raise ValueError(
                        "row-paged rerank requires an on-disk quantized "
                        "target LM head")
                from .lm_head_stream import open_verified_exact_lm_head

                exact_rows = open_verified_exact_lm_head(
                    self._model_dir,
                    self.rc.rerank_lm_head_source,
                    self.rc.rerank_lm_head_source_fingerprint,
                )
                if (exact_rows.vocab != self.cfg.vocab_size
                        or exact_rows.hidden != self.cfg.hidden_size):
                    exact_rows.close()
                    raise ValueError(
                        "exact LM-head source shape does not match target config")
                try:
                    self._lm_head_w = make_row_paged_reranked_q_head(
                        self._lm_head_w, exact_rows,
                        candidates=self.rc.rerank_lm_head_candidates,
                        recall_probe_every=(
                            self.rc.rerank_lm_head_recall_probe_every))
                except Exception:
                    exact_rows.close()
                    raise
                self._reranked_lm_head_source_fingerprint = (
                    self.rc.rerank_lm_head_source_fingerprint)
            else:
                if isinstance(self._lm_head_w, QTensor):
                    raise ValueError(
                        "resident rerank requires an unquantized exact LM head")
                self._reranked_lm_head_exact_resident_bytes = (
                    self._lm_head_w.nbytes)
                self._lm_head_w = make_reranked_q_head(
                    self._lm_head_w,
                    candidates=self.rc.rerank_lm_head_candidates,
                    group_size=self.rc.rerank_lm_head_group_size,
                    bits=self.rc.rerank_lm_head_bits,
                    mode=self.rc.rerank_lm_head_mode,
                )
            self._reranked_lm_head_bytes = self._lm_head_w.approx.nbytes
            if self.rc.rerank_lm_head_rank_capture_path:
                from .lm_head_recall_capture import (
                    AuthoritativeRankCapture,
                    quantized_lm_head_artifact_identity,
                )

                if not self._reranked_lm_head_source_fingerprint:
                    raise ValueError(
                        "rank capture requires a row-paged exact BF16 source")
                approximate_identity = quantized_lm_head_artifact_identity(
                    self._model_dir)
                self._reranked_lm_head_approx_fingerprint = (
                    approximate_identity["fingerprint"])
                self._lm_head_w.recall_rank_capture = AuthoritativeRankCapture(
                    self.rc.rerank_lm_head_rank_capture_path,
                    exact_source_fingerprint=(
                        self._reranked_lm_head_source_fingerprint),
                    approximate_artifact_fingerprint=(
                        self._reranked_lm_head_approx_fingerprint),
                    approximate_artifact_bytes=approximate_identity["bytes"],
                    candidates=self.rc.rerank_lm_head_candidates,
                    vocab=self.cfg.vocab_size,
                    max_positions=(
                        self.rc.rerank_lm_head_rank_capture_max_positions),
                    max_positions_per_request=(
                        self.rc.rerank_lm_head_rank_capture_max_per_request),
                )
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
        pin_first = self.rc.pin_first_layers
        self.planned_trunk_pin_layers = 0
        self.planned_trunk_pin_bytes = 0
        if self.rc.pin_trunk_budget_mb > 0:
            pin_first = self._plan_trunk_pin_layers(n)
            self.planned_trunk_pin_layers = pin_first
        pinned_layers = set(range(pin_first)) | set(
            range(n - self.rc.pin_last_layers, n)
        )
        for i in sorted(pinned_layers):
            # _layer_names: for MoE models this pins attention/norms/router only —
            # experts page separately (pinning all experts would defeat the point)
            for key, names in self._trunk_pages(i):
                self.cache.pin(key, names)

        self.expert_usage: dict[tuple[int, int], int] = {}
        self.expert_hits = 0
        self.expert_misses = 0
        self.expert_trace: list[tuple[int, tuple[int, ...]]] = []  # (layer, routed ids) per fetch, in sweep order
        # Kept parallel to ``expert_trace`` so an explicit offline trace sink
        # can discard prompt-prefill routes and optimize decode placement
        # without changing the long-standing public trace tuple schema.
        self.expert_trace_phases: list[str] = []
        self.expert_route_overlap_trace: list[dict] = []
        self._expert_route_last_by_layer: dict[int, tuple[int, ...]] = {}
        self._expert_route_overlap_totals: dict[str, int] = {}
        self._qwen4_serial_verify_union_layers = 0
        self._qwen4_serial_verify_expert_slots = 0
        self._qwen4_serial_verify_union_experts = 0
        self._qwen4_serial_verify_expert_pages_avoided = 0
        self._qwen4_serial_verify_union_fetch_s = 0.0
        self._qwen4_serial_verify_page_prepare_s = 0.0
        self._qwen4_serial_verify_weight_wait_s = 0.0
        self._qwen4_serial_verify_reserve_s = 0.0
        self._qwen4_serial_verify_linear_compute_s = 0.0
        self._qwen4_serial_verify_full_compute_s = 0.0
        self._qwen4_serial_verify_linear_layers = 0
        self._qwen4_serial_verify_full_layers = 0
        self._qwen4_serial_verify_head_s = 0.0
        self._qwen4_serial_verify_pipelined_expert_layers = 0
        self._expert_compute_batches = 0
        self._max_experts_per_compute_batch = 0
        self._adaptive_expert_batch_clamps = 0
        self._min_adaptive_expert_batch = 0
        self._expert_batch_prefetch_submitted = 0
        self._expert_batch_prefetch_wait_s = 0.0
        self._expert_batch_prefetch_hidden_s = 0.0
        self._expert_batch_prefetch_max_futures = 0
        self._expert_batch_prefetch_phase = "prefill"
        self._expert_batch_prefetch_submitted_by_phase = {
            "prefill": 0, "decode": 0,
        }
        self._expert_batch_prefetch_wait_s_by_phase = {
            "prefill": 0.0, "decode": 0.0,
        }
        self._expert_batch_prefetch_hidden_s_by_phase = {
            "prefill": 0.0, "decode": 0.0,
        }
        self._expert_shared_overlap_layers = 0
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
        if qwen_yarn_factor > 1 and self.cfg.model_type not in ("qwen2", "olmoe"):
            raise ValueError(
                "qwen_yarn_factor is supported only for Qwen2 and OLMoE checkpoints")
        if self.cfg.model_type in ("qwen2", "olmoe"):
            # F94 (2026-07-21): extended from Qwen2-only to also cover OLMoE.
            # yarn_parameters()/supported_qwen_rope_type() were already
            # architecture-agnostic (pure functions over head_dim/rope_theta/
            # scaling dict, tests/test_yarn_parameters.py has no model_type
            # dependency at all) and self._rope_freqs/_mscale are consumed
            # generically by layer_runner.py's run_block for every model
            # that reaches it (OLMoE included -- gpt_oss's own YaRN,
            # _gptoss_rope_state below, already proves a second model type
            # plugging into this same mechanism). The only actual gate was
            # this one explicit model_type check; OLMoE's real config.json
            # confirms standard (non-partial, non-M-RoPE) rotate-half RoPE
            # with rope_scaling=None (no native YaRN, same as an
            # unscaled Qwen2 checkpoint), so it takes the identical
            # qwen_yarn_factor > 1 opt-in extension path.
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
                yarn_label = self.cfg.model_type  # "qwen2" or "olmoe"
                if qwen_yarn_factor > 1:
                    self.effective_max_position_embeddings = int(original * factor)
                    self.rope_profile = f"experimental-{yarn_label}-yarn-{factor:g}x"
                else:
                    self.effective_max_position_embeddings = max(
                        int(self.cfg.max_position_embeddings), int(original * factor))
                    self.rope_profile = f"checkpoint-{yarn_label}-yarn-{factor:g}x"
                self.rope_cache_identity = (
                    f"{yarn_label}-yarn-v1:"
                    f"factor={factor.hex()}:original={original}:"
                    f"beta_fast={beta_fast.hex()}:beta_slow={beta_slow.hex()}:"
                    f"mscale={mscale.hex()}:mscale_all_dim={mscale_all_dim.hex()}"
                )
        if self.cfg.model_type == "gpt_oss":
            self._rope_freqs, self._mscale = _gptoss_rope_state(
                self.cfg, packed=self.store.packed)
            mx.eval(self._rope_freqs)
            # The checkpoint's truncate:false YaRN parameters are reproduced
            # in inverse-frequency space and oracle-checked against the
            # released Transformers implementation.  Bump the cache identity
            # so no endpoint created by either earlier noncanonical formula is
            # ever restored under the corrected positional geometry.
            self.rope_profile = "checkpoint-gptoss-yarn-32x"
            self.rope_cache_identity = "checkpoint-gptoss-yarn-reference-v3"
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
        # Separate from the trunk prefetch queue: this worker carries only the
        # next authoritative routed batches and is awaited before each batch can
        # execute. A persistent single worker avoids per-layer thread startup;
        # depth controls queued futures, not concurrent disk readers.
        self._expert_batch_executor = (
            cf.ThreadPoolExecutor(
                max_workers=max(1, int(getattr(
                    self.rc, "expert_batch_prefetch_workers", 1) or 1)),
                thread_name_prefix="vmodel-expert-batch")
            if self.rc.expert_batch_prefetch and self.cfg.num_experts
            else None
        )
        self._expert_batch_prefetch_active = (
            self._expert_batch_executor is not None)

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
                        self._expert_cache_key(l, e),
                        self.store.names_with_prefix(
                            f"model.layers.{l}.{self.cfg.moe_expert_prefix}.{e}."),
                    )

        # F16: memory-pressure governor — sheds prefetch, MLX scratch, then cache
        # budget when the SYSTEM (not just our budget) runs short. Default on.
        self.governor = None
        if self.rc.governor:
            from .pressure import MemoryGovernor

            governor_kwargs = {}
            system_floor = max(0, int(
                self.rc.hot_prompt_kv_min_available_mb)) * 1_000_000
            if system_floor:
                # Keep the synchronous hot-state admission floor and the
                # background/live allocation ceiling consistent.  Otherwise
                # a long prefill can cross an operator's abort threshold before
                # the next hot-KV admission ever consults that floor.
                governor_kwargs["critical_available"] = system_floor
            self.governor = MemoryGovernor(
                self.cache,
                self.prefetcher,
                floor_bytes=min(
                    self.cache.max_bytes,
                    self.rc.min_weight_cache_mb * 1_000_000,
                ),
                metal_limit=(
                    self.rc.metal_limit_mb * 1_000_000
                    if self.rc.metal_limit_mb else None),
                **governor_kwargs,
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
            mixed_depth_persistence = bool(
                self.cfg.model_type in ("qwen3_5", "qwen3_5_moe")
                and self.rc.qwen_lossy_suffix_prefill_early_layers)
            if mixed_depth_persistence:
                from .qwen_mixed_depth_kv_persist import (
                    QwenMixedDepthPromptPersistence,
                )
            else:
                from .hot_kv_persist import HotPromptKVPersistence

            paged_cache_factory = None
            if self.rc.paged_kv_persist:
                from .kv_paged import PagedKVCache

                def paged_cache_factory(num_layers):
                    cache = PagedKVCache(
                        num_layers,
                        max_bytes=self.rc.max_kv_mb * 1_000_000,
                        spill_dir=self.rc.kv_spill_dir,
                        page_positions=self.rc.kv_page_positions,
                        compress_spill=self.rc.kv_spill_compress,
                    )
                    cache.online_attention = bool(
                        self.cfg.model_type in ("qwen3_5", "qwen3_5_moe")
                        and self.rc.qwen35_paged_online_attention)
                    cache.online_attention_tile_positions = int(
                        self.rc.qwen35_paged_online_tile_positions)
                    cache.online_attention_page_native = bool(
                        self.rc.qwen35_paged_online_page_native)
                    cache.online_attention_pages_per_tile = int(
                        self.rc.qwen35_paged_online_tile_positions
                        // self.rc.kv_page_positions)
                    return cache

            if mixed_depth_persistence:
                if paged_cache_factory is not None:
                    raise ValueError(
                        "mixed-depth Qwen persistence requires resident KV")
                self._hot_kv_persist = QwenMixedDepthPromptPersistence(
                    self.rc.hot_prompt_kv_persist_dir,
                    self._get_kv_fingerprint(),
                    self.rc.hot_prompt_kv_chunk_size,
                    max_checkpoints=(
                        self.rc.hot_prompt_kv_persist_max_checkpoints),
                    max_bytes=(
                        self.rc.hot_prompt_kv_persist_max_mb * 1_000_000),
                    config=self.cfg,
                    allow_prompt_endpoint=(
                        self.rc.qwen_mixed_depth_endpoint_persist),
                )
            else:
                self._hot_kv_persist = HotPromptKVPersistence(
                    self.rc.hot_prompt_kv_persist_dir,
                    self._get_kv_fingerprint(),
                    self.rc.hot_prompt_kv_chunk_size,
                    max_checkpoints=self.rc.hot_prompt_kv_persist_max_checkpoints,
                    max_bytes=self.rc.hot_prompt_kv_persist_max_mb * 1_000_000,
                    config=self.cfg,
                    require_dsa=(
                        self.cfg.model_type == "glm_moe_dsa"
                        and bool(self.cfg.index_topk)
                        and not self._dsa_elided),
                    require_recurrent=(
                        self.cfg.model_type in (
                            "kimi_linear", "kimi_k3", "qwen3_5_moe",
                            "qwen3_5", "qwen4_exp")),
                    paged_cache_factory=paged_cache_factory,
                )
            if not self._defer_persisted_kv_until_bootstrap:
                for (tokens, kv, logits, prompt_length, prompt_logits,
                     reusable_prefix, approximate, tool_capsules,
                     segment_chain, persisted_namespace, exact_hidden
                     ) in self._hot_kv_persist.load_all(
                        self.cfg.num_hidden_layers, self.rc.hot_prompt_kv_slots):
                    self._hot_prompt_slots.append(_HotPromptSlot(
                        tokens=tokens, kv=kv, logits=logits,
                        prompt_length=prompt_length, prompt_logits=prompt_logits,
                        reusable_prefix=reusable_prefix,
                        # F95: durable persistence bakes ONE chunk size into
                        # its on-disk format for the whole store (see
                        # HotPromptKVPersistence.__init__) -- restored slots
                        # always use that fixed, engine-wide value, never
                        # the per-conversation adaptive pick.
                        chunk_size=self.rc.hot_prompt_kv_chunk_size,
                        exact_hidden=exact_hidden,
                        approximate=approximate,
                        tool_capsules=tool_capsules,
                        segment_chain=segment_chain,
                        cache_namespace=persisted_namespace,
                        persisted_preload=True,
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

    def _project_dense_text_kv_bytes(
            self, positions: int, *, stable_boundary_positions: int | None = None
    ) -> int:
        if self.cfg.model_type == "qwen4_exp":
            full_layers = sum(
                kind == "full_attention" for kind in self.cfg.layer_types)
            linear_layers = self.cfg.num_hidden_layers - full_layers
            attention = (
                positions * full_layers * 2
                * self.cfg.num_key_value_heads * self.cfg.head_dim * 2)
            recurrent = (
                linear_layers * self.cfg.linear_num_value_heads
                * self.cfg.linear_key_head_dim
                * self.cfg.linear_value_head_dim * 4)
            conv_width = (
                2 * self.cfg.linear_num_key_heads
                * self.cfg.linear_key_head_dim
                + self.cfg.linear_num_value_heads
                * self.cfg.linear_value_head_dim)
            recurrent_conv = (
                linear_layers
                * max(0, self.cfg.linear_conv_kernel_dim - 1)
                * conv_width * 2)
            qsa_aux = positions * full_layers * (
                self.cfg.qwen4_indexer_kv_heads
                * self.cfg.qwen4_indexer_head_dim * 2 + 4)
            ple_conv = (
                len(self.cfg.qwen4_ple_layers)
                * max(0, self.cfg.qwen4_ple_conv_kernel_size - 1)
                * self.cfg.qwen4_ngram_size
                * self.cfg.qwen4_hc_count * self.cfg.hidden_size * 2)
            return attention + recurrent + recurrent_conv + qsa_aux + ple_conv
        if self.cfg.model_type in ("qwen3_5_moe", "qwen3_5"):
            layer_types = tuple(self.cfg.layer_types)
            full_layers = sum(
                layer_type == "full_attention" for layer_type in layer_types)
            attention_positions = positions * full_layers
            early_layers = int(
                self.rc.qwen_lossy_suffix_prefill_early_layers or 0)
            retained_tokens = (
                int(self.rc.qwen_lossy_suffix_prefill_prefix_tokens or 0)
                + int(self.rc.qwen_lossy_suffix_prefill_tokens or 0))
            if (0 < early_layers < int(self.cfg.num_hidden_layers)
                    and retained_tokens > 0 and positions > retained_tokens):
                # The mixed-depth schedule sends every position through the
                # shallow blocks, but only the configured prefix/suffix plus
                # any generation scaffold through the deep blocks.  Admission
                # must project that physical KV layout.  Charging every full-
                # attention layer for the entire prompt overstates a 30K
                # Huihui request by ~2.8 GB and rejects a layout that the
                # unchanged memory governor can safely hold.
                shallow_full_layers = sum(
                    layer_type == "full_attention"
                    for layer_type in layer_types[:early_layers])
                deep_full_layers = full_layers - shallow_full_layers
                stable = (
                    positions if stable_boundary_positions is None
                    else max(0, min(
                        positions, int(stable_boundary_positions))))
                deep_positions = min(stable, retained_tokens) + (
                    positions - stable)
                attention_positions = (
                    positions * shallow_full_layers
                    + deep_positions * deep_full_layers)
            attention = (
                attention_positions * 2
                * int(self.cfg.num_key_value_heads)
                * int(self.cfg.head_dim) * 4)
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
                * conv_width * 4)
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

        # Match per-layer admission's phase-aware F42 pad.  At construction
        # this is still 400 MB (unmeasured work fails closed); after a complete
        # one-position decode sweep it is zero because ``transient`` is already
        # that sweep's measured maximum and the governor independently keeps
        # its critical system reserve.  Keeping a hard-coded 400 MB here made
        # hot-state admission disagree with the layer allocations it was
        # admitting: a live Qwen3.6-35B-A3B gateway request fit with the learned
        # transient plus the 1.2 GB critical reserve, then was rejected solely
        # by this second copy of the estimator pad.
        margin = int(getattr(self, "_layer_transient_margin", 400_000_000))
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

    def _profile_layer_type(self, i: int) -> str:
        """Stable architecture label for profiler rows."""
        if i < len(self.cfg.layer_types):
            return str(self.cfg.layer_types[i])
        if i in self.cfg.kda_layers:
            return "kda"
        if i in self.cfg.full_attn_layers:
            return "full_attention"
        if i < len(self.cfg.indexer_types):
            return f"mla_{self.cfg.indexer_types[i]}"
        if self.cfg.model_type in (
                "glm_moe_dsa", "kimi_k25", "glm4_moe_lite"):
            return "mla"
        return "attention"

    def _transient_layer_signature(self, i: int) -> str:
        """Architecture-level compute class used for scratch admission.

        Attention implementations and dense/routed MLPs have different
        allocation shapes even when they are adjacent in one model.  The
        signature intentionally contains no model name, prompt property, or
        layer number: observations generalize to every layer with the same
        operations without leaking an outlier across unlike operations.
        """
        attention = self._profile_layer_type(i)
        if not self.cfg.num_experts:
            mlp = "dense"
        elif i < len(self.cfg.mlp_layer_types):
            mlp = str(self.cfg.mlp_layer_types[i])
        else:
            mlp = (
                "dense"
                if i < self.cfg.first_k_dense_replace
                else "moe"
            )
        return f"{attention}+{mlp}"

    def _select_layer_transient(self, position_count: int, layer: int) -> int:
        """Select the learned scratch reserve for one imminent layer."""
        if position_count <= 0:
            raise ValueError("position_count must be positive")
        signature = self._transient_layer_signature(layer)
        learned = int(getattr(
            self, "_layer_transient_by_signature", {}
        ).get((position_count, signature), 0))
        self._layer_transient = max(0, learned)
        self._layer_transient_margin = _layer_transient_reserve_margin(
            position_count)
        return self._layer_transient

    def _record_layer_transient(
            self, position_count: int, layer: int, measured_bytes: int,
    ) -> int:
        """Record one layer and retain both typed and aggregate high-waters."""
        signature = self._transient_layer_signature(layer)
        by_signature = self._layer_transient_by_signature
        key = (position_count, signature)
        measured = max(0, int(measured_bytes))
        counts = getattr(
            self, "_layer_transient_observation_counts", None
        )
        if counts is None:
            counts = {}
            self._layer_transient_observation_counts = counts
        recurring = getattr(
            self, "_layer_transient_recurring_max", None
        )
        if recurring is None:
            recurring = {}
            self._layer_transient_recurring_max = recurring
        count = int(counts.get(key, 0))
        if count == 0:
            learned = measured
        else:
            learned = max(int(recurring.get(key, 0)), measured)
            recurring[key] = learned
        counts[key] = count + 1
        by_signature[key] = learned
        self._layer_transient = learned
        if position_count == 1:
            self._decode_layer_transient = max(
                int(self._decode_layer_transient), learned)
        else:
            by_positions = self._prefill_layer_transient_by_positions
            by_positions[position_count] = max(
                int(by_positions.get(position_count, 0)), learned)
            self._prefill_layer_transient = max(by_positions.values())
        return learned

    def _select_serial_verify_layer_transient(
        self, verifier_positions: int, layer: int
    ) -> int:
        """Select scratch learned only from the same verifier-window width."""
        width = int(verifier_positions)
        if width <= 1:
            return self._select_layer_transient(1, layer)
        signature = self._transient_layer_signature(layer)
        learned = self._serial_verify_layer_transient.get(
            (width, signature)
        )
        if learned is None:
            learned = self._layer_transient_by_signature.get(
                (1, signature), 0
            )
        self._layer_transient = max(0, int(learned))
        self._layer_transient_margin = _layer_transient_reserve_margin(1)
        return self._layer_transient

    def _record_serial_verify_layer_transient(
        self, verifier_positions: int, layer: int, measured_bytes: int
    ) -> int:
        """Record verifier scratch without polluting ordinary decode state."""
        width = int(verifier_positions)
        if width <= 1:
            return self._record_layer_transient(
                1, layer, measured_bytes
            )
        key = (width, self._transient_layer_signature(layer))
        measured = max(0, int(measured_bytes))
        count = int(self._serial_verify_layer_transient_counts.get(key, 0))
        if count == 0:
            learned = measured
        else:
            learned = max(
                int(self._serial_verify_layer_transient_recurring_max.get(
                    key, 0
                )),
                measured,
            )
            self._serial_verify_layer_transient_recurring_max[key] = learned
        self._serial_verify_layer_transient_counts[key] = count + 1
        self._serial_verify_layer_transient[key] = learned
        self._layer_transient = learned
        return learned

    def _restore_aggregate_layer_transient(self, position_count: int) -> int:
        """Restore the request-level high-water after a typed layer loop."""
        (self._layer_transient,
         self._layer_transient_margin) = _layer_transient_for_positions(
             position_count,
             getattr(
                 self, "_prefill_layer_transient_by_positions", {}
             ).get(position_count, 0),
             getattr(self, "_decode_layer_transient", 0))
        return self._layer_transient

    def _layer_names(self, i: int) -> list[str]:
        """Names for the always-needed part of a layer. For MoE layers this is
        attention + norms + router — experts page separately, after routing."""
        names = self.store.layer_param_names(i)
        if self.cfg.num_experts:
            expert_marker = f".{self.cfg.moe_expert_prefix}."
            names = [n for n in names if expert_marker not in n]
        if self.cfg.model_type == "qwen4_exp":
            # PLE's 128 embedding bodies are served by the authenticated
            # direct-row provider. The small learned projection/norm/conv
            # tensors remain in this layer page; table metadata and giant row
            # bodies must never enter the ordinary WeightCache.
            names = [
                name for name in names
                if (
                    ".ple.ple_embedding." not in name
                    and ".ple.ngram_embedding.shards." not in name
                )
            ]
        if self._dsa_elided:
            # F43: with S bounded <= index_topk the indexer selects every position
            # by construction — its weights can never affect output. Skip the bytes.
            names = [n for n in names if ".self_attn.indexer." not in n]
        if self.cfg.model_type == "deepseek_v4":
            # DeepSeek V4's Indexer is not implemented: deepseek_v4_attention
            # never calls index_scores/index_topk_idxs, and the ratio-4 layers
            # use the plain compressed gather instead. Its weights are 7.8% of
            # every trunk layer -- 13.11MB read AND dequantized per layer per
            # token to be discarded.
            #
            # This is elision by non-implementation, not by proof, so it is
            # deliberately fail-loud rather than silent: the names are absent
            # from the page, so if the Indexer path is ever wired up it raises
            # a KeyError at the tensor it needs instead of quietly attending
            # over a subset.
            if not self._dsv4_indexer:
                names = [n for n in names if ".attn.indexer." not in n]
        return names

    def _k3_nf12_split_layer_names(
        self, layer: int
    ) -> tuple[list[str], list[str]]:
        """Partition K3 trunk weights at the exact attention/MLP lifetime."""
        prefix = f"model.layers.{layer}."
        attention = []
        mlp = []
        for name in self._layer_names(layer):
            suffix = name.removeprefix(prefix)
            if (
                suffix.startswith("self_attn.")
                or suffix == "input_layernorm.weight"
                or suffix.startswith("self_attention_res_")
            ):
                attention.append(name)
            else:
                mlp.append(name)
        if not attention or not mlp:
            raise RuntimeError(
                f"layer {layer}: invalid K3 NF12 attention/MLP split "
                f"({len(attention)}/{len(mlp)})"
            )
        return attention, mlp

    def _layer_fetch_bytes_estimate(
        self, layer: int, names: list[str] | None = None
    ) -> int:
        """Conservative materialized trunk-page estimate for pre-fetch eviction.

        Standard MLX quantization retains its physical weight/scale/bias arrays
        directly, so safetensors metadata prices that representation exactly.
        K2.5's checkpoint stores its trunk/router/shared-expert tensors as BF16;
        only routed experts use compressed-tensors INT4 and they have their own
        lifetime-bounded fetch path. Other packed/fused formats remain on their
        architecture-specific estimates because their load representation may
        differ from their logical shapes.
        """
        store = getattr(self, "store", None)
        nf12 = getattr(store, "bf16_nf12_sidecar", None)
        if nf12 is not None and nf12.has_layer(layer):
            entry = nf12.layer_entry(layer)
            # Admission must use the decoded cache representation, not the
            # smaller file stream. ``all_bf16_raw_bytes`` also covers tensors
            # omitted from NF12 as raw fallbacks; pad small controls by 5%.
            if names is None:
                decoded_bytes = int(entry["all_bf16_raw_bytes"])
            else:
                specs = {
                    tensor["name"]: tensor
                    for tensor in entry["tensors"]
                }
                if self.store.bf16_nf12_direct_linear:
                    from .bf16_nf12_linear import (
                        direct_linear_eligible,
                    )

                    decoded_bytes = sum(
                        int(
                            specs[name][
                                "encoded_bytes"
                                if direct_linear_eligible(specs[name])
                                else "raw_bytes"
                            ]
                        )
                        for name in names
                        if name in specs
                    )
                else:
                    decoded_bytes = sum(
                        int(specs[name]["raw_bytes"])
                        for name in names
                        if name in specs
                    )
                # Small raw fallbacks/controls are not represented in the
                # NF12 manifest's selected tensor list.
                decoded_bytes += max(
                    0,
                    int(entry["all_bf16_raw_bytes"])
                    - int(entry["selected_raw_bytes"]),
                )
            return math.ceil(decoded_bytes * 1.05)
        if getattr(store, "on_disk_quantized", False):
            logical_names = (
                self._layer_names(layer) if names is None else names)
            resident_bytes = store.mlx_quantized_resident_bytes(
                logical_names)
            if resident_bytes > 0:
                # Payload bytes are exact for the retained QTensor arrays;
                # keep a small pad for container/alignment accounting.
                return math.ceil(resident_bytes * 1.05)
        # GLM-5.3-Flash is ``glm5_next`` while full GLM-5.3 intentionally
        # keeps the GLM-5.2 ``glm_moe_dsa`` architecture id. Qwen4-Exp FP8
        # derivatives share the same weight+scale representation. Detect it
        # from the store so BF16 releases retain their existing path and every
        # FP8 page is priced at its widened BF16 lifetime.
        if getattr(store, "_glm53_fp8_aux", None):
            logical_names = (
                self._layer_names(layer) if names is None else names)
            resident_bytes = store.finegrained_fp8_resident_bytes(
                logical_names)
            if resident_bytes <= 0:
                raise ValueError(
                    f"layer {layer} has incomplete fine-grained FP8 resident "
                    "metadata")
            return math.ceil(resident_bytes * 1.05)
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

    def _prepare_serial_verify_layer_page(self, layer: int) -> int:
        """Admit a verifier layer before fetching it into Metal memory.

        The ordinary sweep performs this exact ordering. The serial verifier
        historically loaded the page first and reserved only its much smaller
        learned compute transient afterward. Under a tight Metal ceiling that
        made a 10-MB scratch declaration inherit a layer page's overshoot and
        repeatedly ratchet the cache budget. Evict within the current budget,
        then reserve the still-unallocated page so worsening pressure refuses
        before the fetch.
        """
        key = self._layer_key(layer)
        if self.cache.contains(key):
            return 0
        incoming = int(self._layer_fetch_bytes_estimate(layer) or 0)
        if incoming <= 0:
            return 0
        cache_prepare_t0 = time.perf_counter()
        self.cache.prepare_for(incoming)
        self._qwen35_serial_verify_cache_prepare_s = float(getattr(
            self, "_qwen35_serial_verify_cache_prepare_s", 0.0
        )) + (time.perf_counter() - cache_prepare_t0)
        if self.governor is not None:
            # The page estimate above already prices every retained QTensor
            # array and carries its own 5% representation/alignment pad.  Do
            # not add reserve()'s generic 400-MB *second* uncertainty charge:
            # serial verification selects the one-position arithmetic margin
            # immediately before this call, and then uses that same learned
            # margin for the compute-transient admission after the page is
            # resident.  Passing it here keeps the two phases on one safety
            # model instead of rejecting a known ~203-MB page while separately
            # preserving hundreds of megabytes for the same operation.
            exact_page_admission = bool(getattr(
                self.rc,
                "qwen35_serial_verify_exact_page_admission",
                False,
            ))
            page_reserve_t0 = time.perf_counter()
            try:
                self.governor.reserve(
                    incoming,
                    margin=(
                        int(self._layer_transient_margin)
                        if exact_page_admission
                        else 400_000_000
                    ),
                    reason="serial-verify-layer-page",
                )
            finally:
                self._qwen35_serial_verify_page_reserve_s = float(getattr(
                    self, "_qwen35_serial_verify_page_reserve_s", 0.0
                )) + (time.perf_counter() - page_reserve_t0)
        return incoming

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
        dspark_prefetch_plan = getattr(
            self, "_dspark_expert_prefetch_plan", None)
        if dspark_prefetch_plan is not None:
            dspark_prefetch_plan.observe_authoritative(
                layer, expert_ids)
        provisional = getattr(self, "_provisional", None)
        if provisional is not None and positions is not None:
            provisional.append((layer, positions))
        for e in expert_ids:
            if provisional is None:
                self.expert_usage[(layer, e)] = self.expert_usage.get((layer, e), 0) + 1
        self.expert_trace.append((layer, tuple(expert_ids)))
        profiler = getattr(self, "_request_profiler", None)
        trace_phases = getattr(self, "expert_trace_phases", None)
        if trace_phases is None:
            trace_phases = self.expert_trace_phases = []
        trace_phases.append(str(
            getattr(profiler, "phase", "unattributed")
            if profiler is not None else "unattributed"
        ))
        if (
            self.rc.expert_route_overlap_telemetry
            and provisional is None
            and positions
        ):
            summary, last_route = expert_route_overlap_summary(
                positions,
                self._expert_route_last_by_layer.get(layer),
            )
            self._expert_route_last_by_layer[layer] = last_route
            trace_entry = {"layer": int(layer), **summary}
            self.expert_route_overlap_trace.append(trace_entry)
            for key, value in summary.items():
                self._expert_route_overlap_totals[key] = (
                    self._expert_route_overlap_totals.get(key, 0)
                    + int(value)
                )
        # Phase 8: learn routing transitions and prefetch next layer's likely experts
        # (F55: during a provisional sweep, observation is deferred to commit)
        if self.predictor is not None:
            if provisional is None:
                self.predictor.observe(layer, expert_ids)
            if self.prefetcher and self.rc.expert_predictive_prefetch:
                for e in self.predictor.predict(layer, expert_ids, top_m=self.cfg.num_experts_per_tok):
                    self.prefetcher.schedule(
                        self._expert_cache_key(layer + 1, e),
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

    def _trunk_pages(self, layer: int) -> list[tuple[str, list[str]]]:
        """Return the exact (key, names) trunk pages this layer's runner asks for.

        Pinning is only useful if it populates the same keys the sweep looks
        up.  K3's NF12 path deliberately splits each layer into ``.attn`` and
        ``.mlp`` pages so the two have independent lifetimes, and a pin under
        the whole-layer key would be resident but never hit -- it would cost
        residency and return nothing, while looking like "pinning does not
        help". Mirror the runner's own eligibility test rather than guessing.
        """
        key = self._layer_key(layer)
        if (self.rc.layer_stationary_prefill
                and self.store.bf16_nf12_sidecar is not None
                and not self.store.bf16_nf12_uncached_reads
                and self.cfg.model_type == "kimi_k3"):
            attention_names, mlp_names = self._k3_nf12_split_layer_names(layer)
            return [(f"{key}.attn", attention_names), (f"{key}.mlp", mlp_names)]
        return [(key, self._layer_names(layer))]

    def _deepseek_v4_attention(self, x, w, prefix, layer, kv, offset):
        """Attention for one DeepSeek V4 layer.

        Window-only layers (``compress_ratio == 0``) are complete. Compressed
        layers additionally need the Compressor state machine and, at ratio 4,
        the Indexer; both are implemented and oracled in runtime/deepseek_v4.py
        but their per-layer prefill/decode state is not yet held here, so they
        fail explicitly rather than attending over an empty compressed region
        -- which would run and silently drop all long-range context.
        """
        from .deepseek_v4 import (deepseek_v4_attention, window_ring_write,
                                  window_topk_idxs, yarn_freqs)

        ratio = (self.cfg.compress_ratios[layer]
                 if layer < len(self.cfg.compress_ratios) else 0)
        window = self.cfg.window_size
        # The engine derives `offset` from the KV cache's own length, but this
        # attention keeps its own ring and never calls kv.update(), so
        # kv.offset stays 0 forever and every decode step arrives with offset
        # 0. The effective position is therefore resolved ONCE PER SWEEP in
        # _sweep and read here -- advancing it in this method incremented it
        # once per layer, so a single decode token ran layer 0 at position 5,
        # layer 1 at 6, ... layer 42 at 47.
        offset = int(getattr(kv, "dsv4_sweep_pos", offset))

        rings = getattr(kv, "dsv4_rings", None)
        if rings is None:
            rings = {}
            kv.dsv4_rings = rings
        head_dim = self.cfg.head_dim
        ring = rings.get(layer)
        if ring is None:
            ring = mx.zeros((x.shape[0], window, head_dim), dtype=x.dtype)

        from .deepseek_v4 import _packed_matmul

        latent = _packed_matmul(x, w[f"{prefix}.attn.wkv.weight"])
        latent = mx.fast.rms_norm(
            latent, w[f"{prefix}.attn.kv_norm.weight"], self.cfg.rms_norm_eps)
        rope_dim = self.cfg.rope_head_dim
        # A compressed layer builds freqs_cis with compress_rope_theta (and
        # YaRN) for EVERYTHING -- query, window kv, and compressed kv alike.
        # Only pure sliding-window layers use the base rope_theta. Using the
        # base theta on compressed layers left the first token correct (small
        # positions, small phase error) and degraded every token after it.
        theta = self.cfg.compress_rope_theta if ratio else self.cfg.rope_theta
        original = self.cfg.compress_original_seq_len if ratio else 0
        cos, sin = yarn_freqs(
            rope_dim, offset + x.shape[1], original, theta,
            self.cfg.compress_rope_factor if ratio else 1.0, 32, 1)
        from .deepseek_v4 import apply_rope_interleaved

        tail = apply_rope_interleaved(
            latent[..., -rope_dim:], cos[offset:], sin[offset:])
        latent = mx.concatenate([latent[..., :-rope_dim], tail], axis=-1)

        record = getattr(kv, "dsv4_verify_record", None)
        if record is not None:
            # Everything needed to redo this layer for a SHORTER prefix, once
            # the accepted length is known. Cheap: the ring is one window of
            # latents and the rest are per-position projections, so the whole
            # record is a few MB across 43 layers -- far less than retaining a
            # full state snapshot per draft position.
            record[layer] = {"ring": ring, "latent": latent,
                             "store": (getattr(kv, "dsv4_compressed", None)
                                       or {}).get(layer),
                             "cstate": None, "kv_proj": None, "sc_proj": None}
        # Multi-position decode gathers [ring | block] and writes the ring
        # only AFTER attention: a slot holds one position, so writing the
        # block first destroys window entries the block's own earlier queries
        # still need. See block_decode_topk_idxs.
        block_decode = offset > 0 and x.shape[1] > 1
        if not block_decode:
            rings[layer] = window_ring_write(ring, latent, offset, window)
        # Materialize the ring. window_ring_write is a concat around the
        # previous ring, so left lazy each decode step's ring holds a reference
        # to its predecessor and the whole chain is retained: measured at
        # +0.52GB per token across 43 layers, monotonically, which is what
        # exhausted memory on tool-schema prompts. The same applies to the
        # compressed store, which is appended to per emission.
        mx.eval(rings[layer])

        # Prefill attends over the raw sequence, decode over the ring. The
        # released Attention.forward passes `kv` (seqlen entries) at
        # start_pos 0 and `self.kv_cache` (window slots) afterwards, and
        # get_window_topk_idxs returns SEQUENCE positions in its start_pos==0
        # branch versus RING slots after. Using the ring for both was
        # accidentally right only while seqlen <= window, and put compressed
        # entries at window+j while the gather list pointed at seqlen+j -- so
        # every compressed layer read unwritten slots.
        kv_all = (latent if offset == 0
                  else (mx.concatenate([ring, latent], axis=1)
                        if block_decode else rings[layer]))
        compressed_offset = (
            latent.shape[1] if offset == 0
            else (window + x.shape[1] if block_decode else window))
        stores = getattr(kv, "dsv4_compressed", None)
        if stores is None:
            stores = {}
            kv.dsv4_compressed = stores
        states = getattr(kv, "dsv4_cstate", None)
        if states is None:
            states = {}
            kv.dsv4_cstate = states
        if ratio and offset:
            # Decode: absorb this position into the partial group and append a
            # compressed entry only on the step that completes it.
            from .deepseek_v4 import CompressorState

            state = states.get(layer)
            if state is None:
                state = CompressorState(
                    ratio, head_dim, batch=x.shape[0], dtype=mx.float32)
                states[layer] = state
            cw = w[f"{prefix}.attn.compressor.wkv.weight"]
            cg = w[f"{prefix}.attn.compressor.wgate.weight"]
            # Project in the weights' own dtype and widen only the RESULT.
            # Upcasting the weights instead allocated a float32 copy of both
            # [1024, 4096] matrices on every decode step for every compressed
            # layer -- about 16.8MB x 41 layers = 0.69GB per token, which is
            # the sawtooth that remained after the ring fix. Pooling still
            # happens in float32, which is what the released module widens for.
            # One step PER POSITION. The recurrence carries a partial group
            # across steps and emits only on the position that completes it,
            # so a block of draft positions cannot be absorbed in one call --
            # doing so silently pools the wrong positions together. Verifying a
            # draft block is the only caller that arrives with more than one.
            kv_proj = (x.astype(cw.dtype) @ cw.T).astype(mx.float32)
            sc_proj = (x.astype(cg.dtype) @ cg.T).astype(mx.float32)
            if record is not None and layer in record:
                from .deepseek_v4 import CompressorState as _CS

                before = _CS.__new__(_CS)
                before.__dict__.update(state.__dict__)
                record[layer].update({"cstate": before, "kv_proj": kv_proj,
                                      "sc_proj": sc_proj,
                                      "ape": w[f"{prefix}.attn.compressor.ape"],
                                      "cnorm": w[f"{prefix}.attn.compressor.norm.weight"],
                                      "ratio": ratio, "theta": theta,
                                      "original": original})
            for step in range(x.shape[1]):
                here = offset + step
                pooled = state.step(
                    kv_proj[:, step:step + 1], sc_proj[:, step:step + 1],
                    here, w[f"{prefix}.attn.compressor.ape"])
                if pooled is None:
                    continue
                pooled = mx.fast.rms_norm(
                    pooled.astype(x.dtype),
                    w[f"{prefix}.attn.compressor.norm.weight"],
                    self.cfg.rms_norm_eps)
                ccos, csin = yarn_freqs(
                    rope_dim, here + 1, original, theta,
                    self.cfg.compress_rope_factor, 32, 1)
                at = here + 1 - ratio
                ctail = apply_rope_interleaved(
                    pooled[..., -rope_dim:], ccos[at:at + 1], csin[at:at + 1])
                pooled = mx.concatenate(
                    [pooled[..., :-rope_dim], ctail], axis=-1)
                stores[layer] = (
                    pooled if layer not in stores
                    else mx.concatenate([stores[layer], pooled], axis=1))
                mx.eval(stores[layer])
            existing = stores.get(layer)
            if existing is not None:
                kv_all = mx.concatenate([kv_all, existing], axis=1)
        elif ratio:
            # Prefill-time compression: pool whole groups from the same hidden
            # states, RoPE them at their own positions (j * ratio) under the
            # compressed theta, and append after the window region so one
            # gather list addresses both.
            from .deepseek_v4 import compress_prefill

            pooled, _leftover = compress_prefill(
                x, w[f"{prefix}.attn.compressor.wkv.weight"],
                w[f"{prefix}.attn.compressor.wgate.weight"],
                w[f"{prefix}.attn.compressor.ape"],
                w[f"{prefix}.attn.compressor.norm.weight"],
                ratio=ratio, head_dim=head_dim,
                norm_eps=self.cfg.rms_norm_eps)
            if pooled is not None:
                ccos, csin = cos, sin
                stride = mx.arange(pooled.shape[1]) * ratio
                ctail = apply_rope_interleaved(
                    pooled[..., -rope_dim:], ccos[stride], csin[stride])
                pooled = mx.concatenate(
                    [pooled[..., :-rope_dim], ctail], axis=-1)
                stores[layer] = pooled
                kv_all = mx.concatenate([kv_all, pooled], axis=1)
            # Seed the decode state with the trailing partial group. Prefill
            # compresses only whole groups; the released module parks the
            # remaining seqlen % ratio positions in kv_state/score_state and
            # decode finishes that group. Starting decode from an empty state
            # made the first decode step's gather ask for one more compressed
            # entry than existed, reading an unwritten slot -- which is why
            # token 1 was correct and everything after it degenerated.
            from .deepseek_v4 import CompressorState

            seed = CompressorState(
                ratio, head_dim, batch=x.shape[0], dtype=mx.float32)
            remainder = x.shape[1] % ratio
            # Replay enough of the tail that the state matches what continuous
            # processing would hold. At ratio 4 the compressor is OVERLAPPING:
            # it keeps the previous complete group alongside the partial one,
            # so seeding only the remainder leaves the overlap half zeroed --
            # and when the prompt divides evenly the remainder is zero and
            # NOTHING was seeded at all. The next chunk then pooled its first
            # group without the overlap context, which is why chunked prefill
            # diverged from a single sweep. Non-overlapping ratios genuinely
            # need only the partial group.
            replay = remainder + (ratio if seed.overlap else 0)
            replay = min(replay, x.shape[1])
            if replay:
                cw = w[f"{prefix}.attn.compressor.wkv.weight"]
                cg = w[f"{prefix}.attn.compressor.wgate.weight"]
                tail = x[:, -replay:]
                kv_tail = (tail.astype(cw.dtype) @ cw.T).astype(mx.float32)
                sc_tail = (tail.astype(cg.dtype) @ cg.T).astype(mx.float32)
                base = x.shape[1] - replay
                for step in range(replay):
                    # Any entry emitted here was already emitted by
                    # compress_prefill above; the replay exists for the state
                    # it leaves behind, not for its output.
                    seed.step(kv_tail[:, step:step + 1],
                              sc_tail[:, step:step + 1], base + step,
                              w[f"{prefix}.attn.compressor.ape"])
            states[layer] = seed

        # ---- Indexer -------------------------------------------------
        # The released model does not attend over the whole compressed
        # region: a separate Indexer, with its OWN compressor (rotate=True,
        # index_head_dim wide), scores every entry and keeps index_topk of
        # them. Attending to all of them is a fidelity gap that widens with
        # the prompt -- 12,805 entries at 51K against a cap of 512 -- and it
        # is the dominant prefill cost at length.
        indexer_topk = None
        if (self._dsv4_indexer and ratio
                and self.cfg.index_topk
                and f"{prefix}.attn.indexer.wq_b.weight" in w):
            from .deepseek_v4 import (apply_rope_interleaved,
                                      compress_prefill, hadamard_transform,
                                      indexer_select)

            iprefix = f"{prefix}.attn.indexer"
            istates = getattr(kv, "dsv4_istate", None)
            if istates is None:
                istates = {}
                kv.dsv4_istate = istates
            istores = getattr(kv, "dsv4_istore", None)
            if istores is None:
                istores = {}
                kv.dsv4_istore = istores
            ihead = self.cfg.index_head_dim

            def _finish(pooled_raw, positions):
                """norm -> RoPE at the group positions -> Hadamard rotate."""
                out = mx.fast.rms_norm(
                    pooled_raw.astype(x.dtype),
                    w[f"{iprefix}.compressor.norm.weight"],
                    self.cfg.rms_norm_eps)
                tail = apply_rope_interleaved(
                    out[..., -rope_dim:], cos[positions], sin[positions])
                out = mx.concatenate([out[..., :-rope_dim], tail], axis=-1)
                return hadamard_transform(out)

            if offset == 0:
                ipooled, _left = compress_prefill(
                    x, w[f"{iprefix}.compressor.wkv.weight"],
                    w[f"{iprefix}.compressor.wgate.weight"],
                    w[f"{iprefix}.compressor.ape"],
                    w[f"{iprefix}.compressor.norm.weight"],
                    ratio=ratio, head_dim=ihead,
                    norm_eps=self.cfg.rms_norm_eps)
                if ipooled is not None:
                    istores[layer] = _finish(
                        ipooled, mx.arange(ipooled.shape[1]) * ratio)
                seed = CompressorState(ratio, ihead, batch=x.shape[0],
                                       dtype=mx.float32)
                replay = (x.shape[1] % ratio) + (ratio if seed.overlap else 0)
                replay = min(replay, x.shape[1])
                if replay:
                    icw = w[f"{iprefix}.compressor.wkv.weight"]
                    icg = w[f"{iprefix}.compressor.wgate.weight"]
                    tail_x = x[:, -replay:]
                    kt = (tail_x.astype(icw.dtype) @ icw.T).astype(mx.float32)
                    st = (tail_x.astype(icg.dtype) @ icg.T).astype(mx.float32)
                    base = x.shape[1] - replay
                    for step in range(replay):
                        seed.step(kt[:, step:step + 1], st[:, step:step + 1],
                                  base + step, w[f"{iprefix}.compressor.ape"])
                istates[layer] = seed
            else:
                state = istates.get(layer)
                if state is None:
                    state = CompressorState(ratio, ihead, batch=x.shape[0],
                                            dtype=mx.float32)
                    istates[layer] = state
                icw = w[f"{iprefix}.compressor.wkv.weight"]
                icg = w[f"{iprefix}.compressor.wgate.weight"]
                kp = (x.astype(icw.dtype) @ icw.T).astype(mx.float32)
                sp = (x.astype(icg.dtype) @ icg.T).astype(mx.float32)
                for step in range(x.shape[1]):
                    here = offset + step
                    out = state.step(kp[:, step:step + 1], sp[:, step:step + 1],
                                     here, w[f"{iprefix}.compressor.ape"])
                    if out is None:
                        continue
                    entry = _finish(out, mx.array([here + 1 - ratio]))
                    istores[layer] = (
                        entry if layer not in istores
                        else mx.concatenate([istores[layer], entry], axis=1))
            ikv = istores.get(layer)
            if ikv is not None and ikv.shape[1]:
                qr = mx.fast.rms_norm(
                    x @ w[f"{prefix}.attn.wq_a.weight"].T,
                    w[f"{prefix}.attn.q_norm.weight"], self.cfg.rms_norm_eps)
                indexer_topk = indexer_select(
                    qr, x, w[f"{iprefix}.wq_b.weight"],
                    w[f"{iprefix}.weights_proj.weight"], ikv,
                    n_heads=self.cfg.index_n_heads, head_dim=ihead,
                    rope_head_dim=rope_dim, cos=cos[offset:], sin=sin[offset:],
                    ratio=ratio, index_topk=self.cfg.index_topk,
                    start_pos=offset, offset=compressed_offset)

        from .deepseek_v4 import gather_indices

        if block_decode:
            from .deepseek_v4 import block_decode_topk_idxs, compress_topk_idxs

            topk = block_decode_topk_idxs(window, x.shape[1], offset)
            if ratio:
                compressed = (indexer_topk if indexer_topk is not None
                              else compress_topk_idxs(
                                  ratio, x.shape[1], offset,
                                  compressed_offset))
                if compressed.shape[1] != topk.shape[1]:
                    # Same broadcast gather_indices performs: the compressed
                    # generator can return one row when every query in the
                    # block sees the same compressed entries.
                    compressed = mx.broadcast_to(
                        compressed, (compressed.shape[0], topk.shape[1],
                                     compressed.shape[2]))
                topk = mx.concatenate([topk, compressed], axis=-1)
        elif indexer_topk is not None:
            from .deepseek_v4 import window_topk_idxs

            windowed = window_topk_idxs(window, x.shape[1], offset)
            if windowed.shape[1] != indexer_topk.shape[1]:
                windowed = mx.broadcast_to(
                    windowed, (windowed.shape[0], indexer_topk.shape[1],
                               windowed.shape[2]))
            topk = mx.concatenate([windowed, indexer_topk], axis=-1)
        else:
            topk = gather_indices(window, ratio, x.shape[1], offset,
                                  compressed_offset)

        weights = {
            f"{prefix}.attn.{name}": w[f"{prefix}.attn.{name}"]
            for name in ("wq_a.weight", "wq_b.weight", "q_norm.weight",
                         "wo_a.weight", "wo_b.weight", "attn_sink")
        }
        renamed = {k.replace(".weight", "").replace(f"{prefix}.attn.",
                                                    f"{prefix}.attn."): v
                   for k, v in weights.items()}
        attended = deepseek_v4_attention(
            x, {f"{prefix}.attn.wq_a": w[f"{prefix}.attn.wq_a.weight"],
                f"{prefix}.attn.wq_b": w[f"{prefix}.attn.wq_b.weight"],
                f"{prefix}.attn.q_norm": w[f"{prefix}.attn.q_norm.weight"],
                f"{prefix}.attn.wo_a": w[f"{prefix}.attn.wo_a.weight"],
                f"{prefix}.attn.wo_b": w[f"{prefix}.attn.wo_b.weight"],
                f"{prefix}.attn.attn_sink": w[f"{prefix}.attn.attn_sink"]},
            f"{prefix}.attn",
            heads=self.cfg.num_attention_heads, head_dim=head_dim,
            rope_head_dim=rope_dim, q_lora_rank=self.cfg.q_lora_rank,
            o_lora_rank=self.cfg.o_lora_rank, n_groups=self.cfg.o_groups,
            norm_eps=self.cfg.rms_norm_eps,
            cos=cos[offset:], sin=sin[offset:],
            # kv_all, NOT rings[layer]. Both branches above assemble the
            # gathered tensor deliberately: prefill uses the full latent
            # followed by the compressed entries, decode uses the 128-slot ring
            # followed by them, and compressed_offset is set to match whichever
            # was built. Passing the bare ring discarded that assembly. Below
            # window_size the two agree by accident -- slot p % 128 == p -- so
            # short prompts looked correct while every prompt past 128 tokens
            # read the wrong slot for every query and collapsed to BOS, and the
            # compressed region was never attended to at any length.
            kv_all=kv_all, topk_idxs=topk)
        if block_decode:
            # Deferred to here for the reason above: the gather needed the ring
            # to still hold the positions PRECEDING this block.
            rings[layer] = window_ring_write(ring, latent, offset, window)
            mx.eval(rings[layer])
        return attended

    def _deepseek_v4_ffn(self, x, w, prefix, layer, module_base=None,
                         position_slice=None):
        """MoE for one DeepSeek V4 layer, with routed experts paged on demand.

        Routing runs on the layer page (the gate is a small dense tensor that
        arrives with the trunk); only the experts a token actually selected are
        fetched, through the same WeightCache path every other MoE model uses.
        The shared expert is unconditional and lives on the layer page too.
        """
        from .deepseek_v4 import expert_swiglu, moe_combine, moe_gate

        gate_weight = w.get(f"{prefix}.ffn.gate.weight")
        if gate_weight is None:
            # Dense layer: shared expert only, no routing.
            flat = x.reshape(-1, x.shape[-1])
            out = expert_swiglu(
                flat, w[f"{prefix}.ffn.shared_experts.w1.weight"],
                w[f"{prefix}.ffn.shared_experts.w2.weight"],
                w[f"{prefix}.ffn.shared_experts.w3.weight"],
                swiglu_limit=self.cfg.swiglu_limit)
            return out.reshape(x.shape).astype(x.dtype)

        tid2eid = w.get(f"{prefix}.ffn.gate.tid2eid")
        hash_indices = None
        if tid2eid is not None:
            # Hash-routed layer: gate.tid2eid maps each token id to its fixed
            # expert set. self._dsv4_input_ids is the current sweep's token
            # ids, set by _sweep before the layer loop.
            ids = getattr(self, "_dsv4_input_ids", None)
            if ids is None:
                raise ValueError(
                    f"layer {layer} is hash-routed but no input ids were "
                    "recorded for this sweep")
            ids = ids.reshape(-1)
            if position_slice is not None:
                # A layer-stationary prefill evaluates the MoE in tiles, so
                # the ids must be narrowed to the tile. Without this the gate
                # is handed the whole sweep's ids against a tile's rows and
                # the shapes simply fail to broadcast -- which is the good
                # case; a silent misalignment would route every token wrongly.
                start, stop = position_slice
                ids = ids[start:stop]
            if ids.shape[0] != x.shape[0] * x.shape[1]:
                raise ValueError(
                    f"layer {layer} hash routing got {ids.shape[0]} token ids "
                    f"for {x.shape[0] * x.shape[1]} positions")
            hash_indices = tid2eid[ids]
        weights, indices = moe_gate(
            x.reshape(-1, x.shape[-1]), gate_weight,
            w.get(f"{prefix}.ffn.gate.bias"),
            topk=self.cfg.num_experts_per_tok,
            score_func="sqrtsoftplus",
            route_scale=self.cfg.routed_scaling_factor_v4,
            hash_indices=hash_indices)

        expert_ids = sorted({int(e) for row in indices.tolist() for e in row})
        self._record_expert_route(layer, expert_ids)

        # Fetch in bounded groups. Reserving the whole routed union at once is
        # what refused a tool-schema prompt: at 256 experts and top-6 the union
        # over a long prompt reaches ~150 experts, and 150 x 50MB is past the
        # Metal ceiling before any compute. Pages are released between groups.
        batch = max(1, self.rc.expert_fetch_batch or len(expert_ids))
        pages: dict[int, dict] = {}
        loaded: list[int] = []

        def routed(expert, rows, scale):
            page = pages[expert]
            base = (f"{module_base or f'model.layers.{layer}'}"
                    f".{self.cfg.moe_expert_prefix}.{expert}")
            return expert_swiglu(
                rows, page[f"{base}.w1.weight"], page[f"{base}.w2.weight"],
                page[f"{base}.w3.weight"],
                swiglu_limit=self.cfg.swiglu_limit, weights=scale)

        def shared(rows):
            return expert_swiglu(
                rows, w[f"{prefix}.ffn.shared_experts.w1.weight"],
                w[f"{prefix}.ffn.shared_experts.w2.weight"],
                w[f"{prefix}.ffn.shared_experts.w3.weight"],
                swiglu_limit=self.cfg.swiglu_limit)

        flat = x.reshape(-1, x.shape[-1])
        out = shared(flat).astype(mx.float32)
        for start in range(0, len(expert_ids), batch):
            group = expert_ids[start:start + batch]
            pages.update(self._fetch_experts(layer, group, module_base))
            loaded.extend(group)
            out = out + moe_combine(
                flat[None], routed, weights, indices, None,
                n_routed_experts=self.cfg.num_experts,
                only_experts=set(group)).reshape(flat.shape).astype(mx.float32)
            # Materialize before advancing. Left lazy, every group's matmuls
            # stay live and the accumulated graph reached 12.36GB across a
            # ~150-expert union -- the reservation was bounded but the
            # resident set was not.
            mx.eval(out)
            for expert in group:
                pages.pop(expert, None)
                if not self._dsv4_expert_retain:
                    scope = (module_base.replace(".", "_") if module_base
                             else f"layer.{layer}")
                    # Dropping every expert immediately is what bounds the
                    # resident set when pages are dequantized bf16 (50.3MB
                    # each, so one sweep is ~17GB and no cache can hold it).
                    # Under the native MXFP4 path a page is 12.6MB and a whole
                    # sweep is ~4.3GB, which a large budget CAN retain -- so
                    # the drop becomes the only thing preventing reuse.
                    self.cache.discard(f"{scope}.expert.{expert}")
        return out.reshape(x.shape).astype(x.dtype)

    def _plan_trunk_pin_layers(self, num_layers: int) -> int:
        """Size the pinned trunk prefix from measured on-disk layer bytes.

        Fails closed to zero -- ordinary eviction, today's behavior -- if any
        layer's storage size cannot be established from checkpoint metadata.
        Planning on a partial size model would silently pin less than the
        budget allows or overcommit the cache, and neither error is visible
        from the outside.

        A pin is permanent for the engine's lifetime, so the reserve it leaves
        behind has to cover the *largest* thing that competes with it, not the
        average. That is one routed expert fetch batch plus the governor's
        transient margin: a plan that ignores it succeeds at startup and then
        fails the request mid-decode when ``reserve()`` refuses an allocation
        it can no longer make room for (measured on real K3: pinning 1.866GB
        pushed a 1.52GB expert batch past the 7.31GB ceiling). Deriving the
        floor here turns that late crash into an up-front refusal to pin.
        """
        from .cache_policy import plan_pinned_prefix

        layer_bytes = []
        for layer in range(num_layers):
            names = [name for _key, page in self._trunk_pages(layer)
                     for name in page]
            if not names or self.store.storage_bytes_unknown(names):
                print(
                    f"[engine] trunk pin planning disabled: layer {layer} "
                    "has no resolvable storage size")
                return 0
            layer_bytes.append(self.store.storage_bytes(names))
        pin_limit = self.rc.pin_trunk_budget_mb * 1_000_000
        persistent_pinned = self.cache.pinned_bytes
        budget = max(0, self.cache.max_bytes - persistent_pinned)
        expert_batch = (self.rc.expert_fetch_batch
                        or self.cfg.num_experts_per_tok or 1)
        expert_reserve = (
            expert_batch * self._expert_fetch_page_bytes
            if self.cfg.num_experts else 0)
        required_reserve = expert_reserve + self._layer_transient_margin
        reserve = max(self.rc.pin_trunk_expert_reserve_mb * 1_000_000,
                      required_reserve)
        count = plan_pinned_prefix(
            layer_bytes,
            budget,
            reserve_bytes=reserve,
            pin_limit_bytes=pin_limit,
            prefetch_depth=self.rc.prefetch_depth,
        )
        self.planned_trunk_pin_bytes = sum(layer_bytes[:count])
        print(
            f"[engine] trunk pin plan: {count}/{num_layers} layers, "
            f"{self.planned_trunk_pin_bytes / 1e9:.3f}GB pinned of "
            f"{sum(layer_bytes) / 1e9:.3f}GB trunk "
            f"(requested trunk cap {pin_limit / 1e9:.3f}GB, "
            f"{persistent_pinned / 1e9:.3f}GB already pinned, "
            f"{budget / 1e9:.3f}GB cache room, prefetch depth "
            f"{self.rc.prefetch_depth}, expert reserve "
            f"{reserve / 1e9:.3f}GB, of which "
            f"{required_reserve / 1e9:.3f}GB is the mandatory expert-batch "
            f"floor)")
        from .cache_policy import prefetch_starvation_warning

        warning = prefetch_starvation_warning(
            persistent_pinned + self.planned_trunk_pin_bytes,
            self.cache.max_bytes,
            max(layer_bytes[count:] or [0]),
            expert_reserve + self._layer_transient_margin,
            self.rc.prefetch_depth)
        if warning:
            print(warning)
        return count

    def _materialize_packed_trunk(self, page: dict) -> dict:
        """Widen a packed trunk page so every consumer sees ordinary arrays.

        The cache keeps these weights in their released FP8 form, which is half
        the size of the bf16 they would otherwise occupy and therefore twice as
        pinnable. The dequant those bytes avoided at fetch time happens here
        instead, once per layer visit. A no-op for every other model, and for
        DeepSeek V4 pages that were fetched dequantized.
        """
        from .deepseek_v4 import PackedFP8

        if not any(isinstance(v, PackedFP8) for v in page.values()):
            return page
        return {name: (value
                       if (isinstance(value, PackedFP8)
                           and self._dsv4_fused_fp8
                           and name.endswith(_DSV4_FUSED_FP8_SUFFIXES))
                       else (value.materialize()
                             if isinstance(value, PackedFP8) else value))
                for name, value in page.items()}

    def _dsv4_rollback(self, kv, accepted: int, offset: int) -> None:
        """Truncate one verification sweep's state to its accepted prefix.

        The sweep fed a whole draft block, so ring slots and compressor state
        now reflect positions the target rejected. Accepted positions are
        genuinely correct -- the drafted token equalled the target's own -- so
        they are kept and only the tail is undone, which is what makes
        verification cheaper than re-feeding.

        Replays rather than snapshots: redoing the ring write and the
        compressor steps for the first `accepted` positions costs a few MB of
        recorded projections, where a per-position state snapshot would cost
        tens of MB per layer.
        """
        from .deepseek_v4 import (CompressorState, apply_rope_interleaved,
                                  window_ring_write, yarn_freqs)

        record = getattr(kv, "dsv4_verify_record", None)
        if not record:
            return
        window = self.cfg.window_size
        rope_dim = self.cfg.rope_head_dim
        rings = kv.dsv4_rings
        stores = getattr(kv, "dsv4_compressed", {})
        states = getattr(kv, "dsv4_cstate", {})

        for layer, saved in record.items():
            latent = saved["latent"][:, :accepted]
            rings[layer] = (
                window_ring_write(saved["ring"], latent, offset, window)
                if accepted else saved["ring"])
            mx.eval(rings[layer])

            if saved["cstate"] is None:
                continue
            state = CompressorState.__new__(CompressorState)
            state.__dict__.update(saved["cstate"].__dict__)
            ratio = saved["ratio"]
            if saved["store"] is None:
                stores.pop(layer, None)
            else:
                stores[layer] = saved["store"]
            for step in range(accepted):
                here = offset + step
                pooled = state.step(saved["kv_proj"][:, step:step + 1],
                                    saved["sc_proj"][:, step:step + 1],
                                    here, saved["ape"])
                if pooled is None:
                    continue
                pooled = mx.fast.rms_norm(
                    pooled.astype(latent.dtype), saved["cnorm"],
                    self.cfg.rms_norm_eps)
                ccos, csin = yarn_freqs(
                    rope_dim, here + 1, saved["original"], saved["theta"],
                    self.cfg.compress_rope_factor, 32, 1)
                at = here + 1 - ratio
                ctail = apply_rope_interleaved(
                    pooled[..., -rope_dim:], ccos[at:at + 1], csin[at:at + 1])
                pooled = mx.concatenate(
                    [pooled[..., :-rope_dim], ctail], axis=-1)
                stores[layer] = (
                    pooled if layer not in stores
                    else mx.concatenate([stores[layer], pooled], axis=1))
                mx.eval(stores[layer])
            states[layer] = state
        kv.dsv4_verify_record = None
        kv.dsv4_pos = offset + accepted


    # ---- DSpark multi-token draft (mtp.* stages) ----------------------

    def _dspark_stage_names(self, stage: int) -> list[str]:
        return [n for n in self.store.names_with_prefix(f"mtp.{stage}.")
                if f".{self.cfg.moe_expert_prefix}." not in n]

    def _dspark_stage_page(self, stage: int) -> dict:
        page = self.cache.get(f"mtp.{stage}.trunk",
                              self._dspark_stage_names(stage))
        if self._dsv4_packed_trunk:
            page = self._materialize_packed_trunk(page)
        return page

    def _dspark_stage_count(self) -> int:
        """Derived from the checkpoint, not config.

        config.json's n_mtp_layers is null and num_nextn_predict_layers is a
        different HF-compat field reading 1, while three complete stages ship.
        """
        count = 0
        while self.store.names_with_prefix(f"mtp.{count}."):
            count += 1
        return count

    def _dspark_main_x(self, stage0_page: dict, main_hidden: mx.array):
        from .dsv4_dspark import dspark_main_x

        return dspark_main_x(
            main_hidden, stage0_page["mtp.0.main_proj.weight"],
            stage0_page["mtp.0.main_norm.weight"],
            norm_eps=self.cfg.rms_norm_eps)

    def _dspark_prefill_rings(self, kv, main_hidden: mx.array) -> None:
        """Fill each draft stage's window ring from the prompt.

        The released DSparkBlock.forward returns x unchanged at start_pos == 0
        and only warms its attention cache, so this runs the KV half of each
        stage over the whole prompt and never its MoE -- cheap, and without it
        the first draft attends over an empty ring.
        """
        from .deepseek_v4 import apply_rope_interleaved, window_ring_write, yarn_freqs

        stages = self._dspark_stage_count()
        if stages == 0:
            return
        pages = [self._dspark_stage_page(i) for i in range(stages)]
        main_x = self._dspark_main_x(pages[0], main_hidden)
        rope_dim = self.cfg.rope_head_dim
        window = self.cfg.window_size
        seq = main_x.shape[1]
        cos, sin = yarn_freqs(rope_dim, seq, 0, self.cfg.rope_theta, 1.0, 32, 1)

        rings = getattr(kv, "dspark_rings", None)
        if rings is None:
            rings = {}
            kv.dspark_rings = rings
        for stage in range(stages):
            page = pages[stage]
            prefix = f"mtp.{stage}.attn"
            kvp = mx.fast.rms_norm(
                main_x.astype(page[f"{prefix}.wkv.weight"].dtype)
                @ page[f"{prefix}.wkv.weight"].T,
                page[f"{prefix}.kv_norm.weight"], self.cfg.rms_norm_eps)
            tail = apply_rope_interleaved(kvp[..., -rope_dim:], cos, sin)
            kvp = mx.concatenate([kvp[..., :-rope_dim], tail], axis=-1)
            ring = rings.get(stage)
            if ring is None:
                ring = mx.zeros((main_x.shape[0], window, self.cfg.head_dim),
                                dtype=kvp.dtype)
            rings[stage] = window_ring_write(ring, kvp, 0, window)
            mx.eval(rings[stage])

    def _dspark_draft(self, kv, main_hidden: mx.array, current_token: int,
                      position: int) -> list[int]:
        """Propose dspark_block_size tokens from one draft pass.

        Costs three stages against the target's 43, and every stage reuses the
        target's own expert paging, so a proposal is roughly 8% of a target
        sweep. Returns [] when drafting is not possible at this position.
        """
        from .dsv4_dspark import (draft_input_ids, dspark_attention,
                                  dspark_sample_block, run_dspark_stage)
        from .deepseek_v4 import hc_head, yarn_freqs

        block = self.cfg.dspark_block_size
        window = self.cfg.window_size
        if position <= 0 or block <= 0:
            return []

        stages = self._dspark_stage_count()
        if stages == 0:
            return []
        pages = [self._dspark_stage_page(i) for i in range(stages)]
        main_x = self._dspark_main_x(pages[0], main_hidden)

        ids = draft_input_ids(current_token, block,
                              self.cfg.dspark_noise_token_id)
        x = self._embed([int(v) for v in ids[0].tolist()])
        x = mx.broadcast_to(x[:, :, None, :],
                            (x.shape[0], x.shape[1], self.cfg.hc_mult,
                             x.shape[2]))

        rope_dim = self.cfg.rope_head_dim
        # Draft stages are window-only (the released DSparkAttention asserts
        # compress_ratio == 0), so the base theta applies, never the
        # compressed one.
        cos, sin = yarn_freqs(rope_dim, position + 1 + block, 0,
                              self.cfg.rope_theta, 1.0, 32, 1)
        rings = getattr(kv, "dspark_rings", None)
        if rings is None:
            rings = {}
            kv.dspark_rings = rings

        for stage in range(stages):
            page = pages[stage]
            prefix = f"mtp.{stage}.attn"
            ring = rings.get(stage)
            if ring is None:
                ring = mx.zeros((x.shape[0], window, self.cfg.head_dim),
                                dtype=x.dtype)

            def attention(t, _p=page, _pre=prefix, _r=ring, _s=stage):
                out, updated = dspark_attention(
                    t, main_x, _p, _pre, ring=_r, start_pos=position,
                    heads=self.cfg.num_attention_heads,
                    head_dim=self.cfg.head_dim, rope_head_dim=rope_dim,
                    q_lora_rank=self.cfg.q_lora_rank,
                    o_lora_rank=self.cfg.o_lora_rank,
                    n_groups=self.cfg.o_groups,
                    norm_eps=self.cfg.rms_norm_eps, window=window,
                    cos=cos[position + 1:], sin=sin[position + 1:],
                    main_cos=cos[position:position + 1],
                    main_sin=sin[position:position + 1])
                rings[_s] = updated
                return out

            hc = {name: page[f"mtp.{stage}.hc_{name}"]
                  for name in ("attn_fn", "attn_scale", "attn_base",
                               "ffn_fn", "ffn_scale", "ffn_base")}
            norms = {"attn": page[f"mtp.{stage}.attn_norm.weight"],
                     "ffn": page[f"mtp.{stage}.ffn_norm.weight"]}
            x = run_dspark_stage(
                x, hc, norms, attention,
                lambda t, _p=page, _s=stage: self._deepseek_v4_ffn(
                    t, _p, f"mtp.{_s}", 0, module_base=f"mtp.{_s}"),
                hc_mult=self.cfg.hc_mult, norm_eps=self.cfg.rms_norm_eps,
                sinkhorn_iters=self.cfg.hc_sinkhorn_iters,
                hc_eps=self.cfg.hc_eps)
            mx.eval(x, rings[stage])

        last = pages[stages - 1]
        base = f"mtp.{stages - 1}"
        reduced = hc_head(x, last[f"{base}.hc_head_fn"],
                          last[f"{base}.hc_head_scale"],
                          last[f"{base}.hc_head_base"],
                          norm_eps=self.cfg.rms_norm_eps,
                          eps=self.cfg.hc_eps)
        normed = mx.fast.rms_norm(reduced, last[f"{base}.norm.weight"],
                                  self.cfg.rms_norm_eps)
        logits = normed.astype(mx.float32) @ self._lm_head_weight().astype(
            mx.float32).T
        drafted, _embeds = dspark_sample_block(
            logits, current_token,
            last[f"{base}.markov_head.markov_w1.weight"],
            last[f"{base}.markov_head.markov_w2.weight"])
        return drafted


    # ---- DeepSeek V4 prefix checkpoint persistence --------------------

    def _dsv4_prefix_dir(self):
        """Directory for on-disk prefix checkpoints, or None."""
        import os as _os
        from pathlib import Path as _Path

        raw = (getattr(getattr(self, "rc", None),
                       "dsv4_prefix_cache_dir", "")
               or _os.environ.get("VMODEL_DSV4_PREFIX_CACHE_DIR", ""))
        if not raw:
            return None
        path = _Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _dsv4_prefix_fingerprint(self) -> str:
        """Identity of everything that changes what a stored prefix MEANS.

        Deliberately NOT _get_kv_fingerprint: that folds in runtime state
        which mutates during a request, so two processes with identical
        configuration produced different values and a written checkpoint was
        never readable back. This is narrow and deterministic -- the
        checkpoint directory, the architecture parameters the state's shape
        and meaning depend on, and the flags that change its arithmetic.
        """
        import hashlib
        import os as _os

        parts = [
            str(getattr(self, "_model_dir", "")),
            str(self.cfg.model_type),
            str(self.cfg.num_hidden_layers),
            str(self.cfg.head_dim),
            str(self.cfg.window_size),
            str(tuple(self.cfg.compress_ratios or ())),
            str(self.cfg.rope_theta),
            str(self.cfg.compress_rope_theta),
            str(self.cfg.index_topk),
            *(f"{name}={_os.environ.get(name, '')}" for name in (
                "VMODEL_DSV4_NATIVE_MXFP4", "VMODEL_DSV4_PACKED_TRUNK",
                "VMODEL_DSV4_FUSED_FP8", "VMODEL_DSV4_CARRIER_DTYPE",
                "VMODEL_DSV4_INDEXER")),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def _dsv4_prefix_write(self, tokens, boundary, per_layer) -> bool:
        """Persist one checkpoint so a COLD start can skip its prefill.

        An agent harness sends the same tool preamble on every request, and
        that preamble is known before any request arrives -- so the 62 minutes
        turn one spends on it is avoidable outright, not merely amortised
        across a session. The payload is dominated by the compressed stores
        (~609MB at a 49,152 boundary); rings and compressor state are ~12MB.

        The token prefix is stored ALONGSIDE the arrays rather than hashed,
        so a load can verify the match exactly instead of trusting a digest.
        """
        import json as _json

        directory = self._dsv4_prefix_dir()
        if directory is None or boundary <= 0:
            return False
        arrays = {"tokens": mx.array(
            [int(t) for t in tokens[:boundary]], dtype=mx.int32)}
        layers = []
        for layer, saved in per_layer.items():
            layers.append(layer)
            if saved["ring"] is not None:
                arrays[f"ring.{layer}"] = saved["ring"]
            if saved["store"] is not None:
                arrays[f"store.{layer}"] = saved["store"]
            state = saved["cstate"]
            if state is not None:
                arrays[f"cstate_kv.{layer}"] = state.kv_state
                arrays[f"cstate_score.{layer}"] = state.score_state
        meta = {
            "fingerprint": self._dsv4_prefix_fingerprint(),
            "boundary": str(boundary),
            "layers": _json.dumps(sorted(layers)),
            "ratio": _json.dumps(
                [int(r) for r in (self.cfg.compress_ratios or ())]),
            "head_dim": str(self.cfg.head_dim),
        }
        target = directory / f"dsv4_prefix_{boundary}.safetensors"
        mx.eval(list(arrays.values()))
        mx.save_safetensors(str(target), arrays, metadata=meta)
        return True

    def _dsv4_prefix_read(self, kv, tokens) -> int:
        """Load the longest on-disk checkpoint that prefixes ``tokens``.

        Same rule as the in-memory path: the whole stored prefix must match,
        because the state covers exactly those positions. A checkpoint from a
        different model or runtime is skipped on its fingerprint rather than
        producing confident nonsense.
        """
        import json as _json

        from .deepseek_v4 import CompressorState

        directory = self._dsv4_prefix_dir()
        if directory is None:
            return 0
        want = self._dsv4_prefix_fingerprint()
        best = None
        for path in sorted(directory.glob("dsv4_prefix_*.safetensors")):
            try:
                arrays, meta = mx.load(str(path), return_metadata=True)
            except Exception:
                continue
            if meta.get("fingerprint") != want:
                continue
            stored = arrays.get("tokens")
            if stored is None:
                continue
            n = int(stored.shape[0])
            if n > len(tokens) or (best is not None and n <= best[0]):
                continue
            head = mx.array([int(t) for t in tokens[:n]], dtype=mx.int32)
            if not bool(mx.all(head == stored.astype(mx.int32))):
                continue
            best = (n, arrays, meta)
        if best is None:
            return 0

        boundary, arrays, meta = best
        rings, stores, states = {}, {}, {}
        ratios = _json.loads(meta.get("ratio") or "[]")
        head_dim = int(meta.get("head_dim") or self.cfg.head_dim)
        for layer in _json.loads(meta.get("layers") or "[]"):
            if f"ring.{layer}" in arrays:
                rings[layer] = arrays[f"ring.{layer}"]
            if f"store.{layer}" in arrays:
                stores[layer] = arrays[f"store.{layer}"]
            if f"cstate_kv.{layer}" in arrays:
                ratio = ratios[layer] if layer < len(ratios) else 0
                state = CompressorState(
                    max(int(ratio), 1), head_dim, batch=1, dtype=mx.float32)
                state.kv_state = arrays[f"cstate_kv.{layer}"]
                state.score_state = arrays[f"cstate_score.{layer}"]
                states[layer] = state
        kv.dsv4_rings = rings
        kv.dsv4_compressed = stores
        kv.dsv4_cstate = states
        kv.dsv4_pos = boundary
        return boundary

    def _dsv4_prefix_store(self, tokens) -> int:
        """Persist the checkpoint the last sweep captured, keyed by its prefix.

        Returns the boundary stored, or 0. The value keeps the ARRAYS the
        sweep produced and copies only the containers, the same ownership rule
        the exact-prompt snapshot uses: decode rebinds entries rather than
        mutating arrays, so sharing arrays is safe and sharing containers is
        not.
        """
        captured = getattr(self, "_dsv4_checkpoint", None)
        self._dsv4_checkpoint = None
        if not captured:
            return 0
        boundary, per_layer = captured
        if boundary <= 0 or boundary > len(tokens):
            return 0
        key = tuple(int(t) for t in tokens[:boundary])
        if key not in self._dsv4_prefix_checkpoints:
            try:
                self._dsv4_prefix_write(tokens, boundary, per_layer)
            except Exception as exc:            # persistence is best-effort
                print(f"[engine] prefix checkpoint not written: {exc}")
        self._dsv4_prefix_checkpoints[key] = {
            "pos": boundary, "layers": per_layer}
        while len(self._dsv4_prefix_checkpoints) > self._dsv4_prefix_slots:
            self._dsv4_prefix_checkpoints.pop(
                next(iter(self._dsv4_prefix_checkpoints)))
        return boundary

    def _dsv4_prefix_restore(self, kv, tokens) -> int:
        """Resume from the longest stored prefix of ``tokens``. Returns its length.

        Only a checkpoint whose ENTIRE key is a prefix of this request may be
        used: its state covers exactly those positions, so a request that
        diverges earlier cannot use it, and one that diverges later resumes
        from it and prefills the remainder.
        """
        from .deepseek_v4 import CompressorState

        if not self._dsv4_prefix_checkpoints:
            return self._dsv4_prefix_read(kv, tokens)
        best = None
        for key, entry in self._dsv4_prefix_checkpoints.items():
            n = len(key)
            if n > len(tokens) or (best is not None and n <= len(best[0])):
                continue
            if all(int(tokens[i]) == key[i] for i in range(n)):
                best = (key, entry)
        if best is None:
            return self._dsv4_prefix_read(kv, tokens)
        entry = best[1]
        rings, stores, states = {}, {}, {}
        for layer, saved in entry["layers"].items():
            if saved["ring"] is not None:
                rings[layer] = saved["ring"]
            if saved["store"] is not None:
                stores[layer] = saved["store"]
            if saved["cstate"] is not None:
                clone = CompressorState.__new__(CompressorState)
                clone.__dict__.update(saved["cstate"].__dict__)
                states[layer] = clone
        kv.dsv4_rings = rings
        kv.dsv4_compressed = stores
        kv.dsv4_cstate = states
        kv.dsv4_pos = entry["pos"]
        return entry["pos"]

    # ---- DeepSeek V4 exact-prompt state reuse -------------------------

    def _dsv4_snapshot_copy(self, kv):
        """Detach a reusable copy of this request's DeepSeek V4 prompt state.

        Decode rebinds dict entries and CompressorState attributes rather than
        mutating arrays in place, so sharing the ARRAYS is safe while sharing
        the CONTAINERS is not: the next request would advance the snapshot's
        own ring. Copy every container, share every array.
        """
        from .deepseek_v4 import CompressorState

        states = {}
        for layer, state in (getattr(kv, "dsv4_cstate", None) or {}).items():
            clone = CompressorState.__new__(CompressorState)
            clone.__dict__.update(state.__dict__)
            states[layer] = clone
        return {
            "rings": dict(getattr(kv, "dsv4_rings", None) or {}),
            "compressed": dict(getattr(kv, "dsv4_compressed", None) or {}),
            "cstate": states,
            "pos": int(getattr(kv, "dsv4_pos", 0)),
        }

    def _dsv4_snapshot_store(self, tokens, kv, logits) -> None:
        if not getattr(kv, "dsv4_rings", None):
            return
        snapshot = self._dsv4_snapshot_copy(kv)
        snapshot["logits"] = logits
        key = tuple(int(t) for t in tokens)
        self._dsv4_snapshots[key] = snapshot
        # Bounded, newest-first. Each entry is ~20MB of state plus one logits
        # row, so a handful is a few hundred MB at most.
        while len(self._dsv4_snapshots) > self._dsv4_snapshot_slots:
            self._dsv4_snapshots.pop(next(iter(self._dsv4_snapshots)))

    def _dsv4_snapshot_lookup(self, tokens):
        """Exact match only.

        A shorter prefix cannot be served from a longer snapshot: the window
        ring holds the LAST 128 positions, so truncating the prompt would need
        state this snapshot no longer contains.
        """
        return self._dsv4_snapshots.get(tuple(int(t) for t in tokens))

    def _dsv4_snapshot_restore(self, kv, snapshot) -> None:
        from .deepseek_v4 import CompressorState

        states = {}
        for layer, state in snapshot["cstate"].items():
            clone = CompressorState.__new__(CompressorState)
            clone.__dict__.update(state.__dict__)
            states[layer] = clone
        kv.dsv4_rings = dict(snapshot["rings"])
        kv.dsv4_compressed = dict(snapshot["compressed"])
        kv.dsv4_cstate = states
        kv.dsv4_pos = snapshot["pos"]

    def _set_finegrained_fp8_direct_phase(self, phase: str) -> None:
        """Select the released expert representation at a safe phase boundary.

        A decode-only direct-QMV profile materializes ordinary BF16 expert
        pages during multi-position prefill, then retains packed E4M3/F32
        pages for singleton decode.  The cache key below includes the active
        representation, so a page produced in one phase can never be consumed
        under the other phase's arithmetic contract.
        """
        if phase not in ("prefill", "decode"):
            raise ValueError(f"unsupported direct-QMV phase {phase!r}")
        self._expert_batch_prefetch_phase = phase
        self._expert_batch_prefetch_active = bool(
            self._expert_batch_executor is not None
            and (
                not self.rc.qwen4_expert_batch_prefetch_prefill_only
                or phase == "prefill"
            )
        )
        store = self.store
        for family in ("glm53", "qwen4"):
            requested = bool(getattr(
                store, f"{family}_fp8_direct_qmv", False))
            if not requested:
                continue
            decode_only = bool(getattr(
                store, f"{family}_fp8_direct_qmv_decode_only", False))
            setattr(
                store,
                f"{family}_fp8_direct_qmv_active",
                bool(not decode_only or phase == "decode"),
            )

    def _expert_cache_key(
        self, layer: int, expert: int, module_base: str | None = None,
    ) -> str:
        scope = module_base.replace(".", "_") if module_base else f"layer.{layer}"
        store = self.store
        family = (
            "qwen4" if getattr(store, "qwen4_fp8_direct_qmv", False)
            else "glm53")
        if getattr(store, f"{family}_fp8_direct_qmv_decode_only", False):
            representation = (
                "fp8direct" if getattr(
                    store, f"{family}_fp8_direct_qmv_active", False)
                else "bf16")
            scope = f"{scope}.{representation}"
        return f"{scope}.expert.{expert}"

    def _fetch_experts(self, layer: int, expert_ids: list[int],
                       module_base: str | None = None) -> dict[int, dict]:
        """Fetch one lifetime-bounded expert batch; routing was recorded already.

        ``module_base`` overrides the ``model.layers.{layer}`` prefix so the
        DSpark draft stages, which live under ``mtp.{stage}``, reuse this exact
        paging path rather than a parallel one.
        """
        items = []
        n_missing = 0
        base = module_base or f"model.layers.{layer}"
        for e in expert_ids:
            key = self._expert_cache_key(layer, e, module_base)
            if self.cache.contains(key):
                self.expert_hits += 1
            else:
                self.expert_misses += 1
                n_missing += 1
            items.append((
                key,
                self.store.names_with_prefix(
                    f"{base}.{self.cfg.moe_expert_prefix}.{e}."),
            ))
        if self.governor is not None and n_missing:
            # Reserve only the pages that can coexist in THIS compute batch.
            # Reserving the full routed union recreates the 16-22 GB false demand
            # even when fetch and compute lifetimes are correctly bounded.
            self.governor.reserve(
                n_missing * self._expert_fetch_page_bytes + self._layer_transient,
                margin=self._layer_transient_margin,
                reason=(
                    "glm53-expert-page"
                    if self.cfg.model_type == "glm5_next"
                    else ""
                ))

        t0 = time.perf_counter()
        pages = self.cache.get_many(items)
        elapsed = time.perf_counter() - t0
        self.timer.add("expert_wait", elapsed)
        profiler = self._request_profiler
        if profiler is not None:
            profiler.record_expert_fetch(
                layer, pages=len(expert_ids), misses=n_missing,
                wall_s=elapsed)
        return {
            e: pages[self._expert_cache_key(layer, e, module_base)]
            for e in expert_ids
        }

    def _get_experts(self, layer: int, expert_ids: list[int],
                     positions: dict[int, list[int]] | None = None) -> dict[int, dict]:
        """Compatibility path: record and return the complete routed union."""
        self._record_expert_route(layer, expert_ids, positions)
        return self._fetch_experts(layer, expert_ids)

    def _iter_expert_batches(self, layer: int, expert_ids: list[int],
                             positions: dict[int, list[int]] | None = None):
        """Return bounded expert pages for immediate compute and release.

        This is F74-v2's actual lifetime boundary. ``WeightCache.get_many`` may
        split disk fetches, but returning a dict for the whole union leaves every
        evicted tensor strongly referenced. The GLM runner consumes one yielded
        mapping, ``mx.eval`` materializes its accumulated output, then advances.

        With ``expert_batch_prefetch``, the authoritative routed union is still
        divided in the same order and at the same live-governor boundary. The
        exact leading fetches are submitted before this method returns, and
        future batches run in order while the current batch computes. The
        explicit depth bounds queued/resident successors; one remains the
        default and previously proven schedule.
        """
        self._record_expert_route(layer, expert_ids, positions)
        position_union = {
            position for expert_positions in (positions or {}).values()
            for position in expert_positions
        }
        single_position = bool(positions) and len(position_union) == 1
        configured_fetch_batch_size = (
            self.rc.decode_expert_fetch_batch
            if single_position and self.rc.decode_expert_fetch_batch > 0
            else self.rc.expert_fetch_batch
        ) or len(expert_ids) or 1
        configured_compute_batch_size = (
            getattr(self.rc, "expert_compute_batch", 0)
            or configured_fetch_batch_size
        )
        governor = getattr(self, "governor", None)

        def plan_batch(start: int) -> tuple[list[int], int]:
            remaining = len(expert_ids) - start
            batch_size = min(configured_fetch_batch_size, remaining)
            if governor is not None and batch_size > 1:
                admitted = governor.admissible_units(
                    unit_bytes=self._expert_fetch_page_bytes,
                    fixed_bytes=self._layer_transient,
                    max_units=batch_size,
                    margin=self._layer_transient_margin,
                )
                if admitted < batch_size:
                    self._adaptive_expert_batch_clamps += 1
                batch_size = admitted
                # Preserve compute/materialization boundaries when a larger
                # storage batch is pressure-clamped. For example, a fetch cap
                # of 32 with a compute boundary of 16 may safely shrink to 16,
                # but not 24: 16+8 would change the floating accumulation
                # grouping relative to the validated 16-wide path.
                if (
                    remaining > configured_compute_batch_size
                    and batch_size >= configured_compute_batch_size
                    and configured_fetch_batch_size
                    > configured_compute_batch_size
                ):
                    batch_size = max(
                        configured_compute_batch_size,
                        (
                            batch_size // configured_compute_batch_size
                        ) * configured_compute_batch_size,
                    )
                self._min_adaptive_expert_batch = (
                    batch_size if self._min_adaptive_expert_batch == 0
                    else min(self._min_adaptive_expert_batch, batch_size)
                )
            batch_ids = expert_ids[start:start + batch_size]
            return batch_ids, start + batch_size

        def compute_chunks(batch_ids: list[int], pages: dict):
            for offset in range(0, len(batch_ids), configured_compute_batch_size):
                compute_ids = batch_ids[
                    offset:offset + configured_compute_batch_size
                ]
                self._expert_compute_batches += 1
                self._max_experts_per_compute_batch = max(
                    self._max_experts_per_compute_batch, len(compute_ids)
                )
                # The producer deliberately retains the complete fetched page
                # mapping across these sub-batches. The consumer materializes
                # and drops each arithmetic group before requesting the next;
                # only after the final group may the next storage batch become
                # authoritative.
                yield compute_ids, pages

        def timed_fetch(batch_ids: list[int]):
            started = time.perf_counter()
            pages = self._fetch_experts(layer, batch_ids)
            return pages, time.perf_counter() - started

        if not expert_ids:
            return iter(())
        executor = getattr(self, "_expert_batch_executor", None)
        if not getattr(
                self, "_expert_batch_prefetch_active", executor is not None):
            executor = None
        if executor is None:
            def synchronous_batches():
                start = 0
                while start < len(expert_ids):
                    batch_ids, start = plan_batch(start)
                    pages = self._fetch_experts(layer, batch_ids)
                    yield from compute_chunks(batch_ids, pages)
                    del pages

            return synchronous_batches()

        prefetch_depth = max(1, int(getattr(
            self.rc, "expert_batch_prefetch_depth", 1) or 1))
        pending = deque()

        def fill_pending(start: int) -> int:
            while len(pending) < prefetch_depth and start < len(expert_ids):
                queued_ids, start = plan_batch(start)
                pending.append((
                    queued_ids, executor.submit(timed_fetch, queued_ids)))
                self._expert_batch_prefetch_submitted += 1
                submitted_by_phase = getattr(
                    self, "_expert_batch_prefetch_submitted_by_phase", None)
                if submitted_by_phase is not None:
                    phase = getattr(
                        self, "_expert_batch_prefetch_phase", "prefill")
                    submitted_by_phase[phase] += 1
                self._expert_batch_prefetch_max_futures = max(
                    self._expert_batch_prefetch_max_futures, len(pending))
            return start

        start = fill_pending(0)

        def pipelined_batches():
            try:
                current_start = start
                while pending:
                    current_ids, future = pending.popleft()
                    current_start = fill_pending(current_start)
                    wait_started = time.perf_counter()
                    pages, fetch_s = future.result()
                    wait_s = time.perf_counter() - wait_started
                    self._expert_batch_prefetch_wait_s += wait_s
                    hidden_s = max(0.0, fetch_s - wait_s)
                    self._expert_batch_prefetch_hidden_s += hidden_s
                    phase = getattr(
                        self, "_expert_batch_prefetch_phase", "prefill")
                    wait_by_phase = getattr(
                        self, "_expert_batch_prefetch_wait_s_by_phase", None)
                    hidden_by_phase = getattr(
                        self, "_expert_batch_prefetch_hidden_s_by_phase", None)
                    if wait_by_phase is not None:
                        wait_by_phase[phase] += wait_s
                    if hidden_by_phase is not None:
                        hidden_by_phase[phase] += hidden_s
                    yield from compute_chunks(current_ids, pages)
                    # The consumer deletes its reference before resuming us.
                    # Drop the producer's reference before advancing.
                    del pages
            finally:
                for _batch_ids, future in pending:
                    future.cancel()
                pending.clear()

        return pipelined_batches()

    def _router_lookahead(self, x: mx.array, nxt: int) -> None:
        """F45 (MoE-SpeQ class; lossless — prefetch is only a cache hint):
        predict layer `nxt`'s routed experts by running its ACTUAL router on
        the current hidden state (routing is largely stable across one block)
        and prefetch that union. Token-conditioned, unlike the Markov
        transition predictor. Never blocks on disk: skips unless the next
        layer's page is already resident, and the default idle-only gate admits
        no backlog behind existing prefetch work.

        F198: the gate/router name is derived from ``moe_expert_prefix`` rather
        than assumed to be ``mlp.``. Kimi Linear and K3 ship their gate under
        ``block_sparse_moe.gate.*``, so a hardcoded ``mlp.`` lookup found
        neither weight, fell through to the dense-layer branch, and returned
        without scheduling anything -- the predictor was silently inert on
        exactly the architectures whose expert paging costs the most.
        """
        page = self._trunk_pages(nxt)[-1]  # the gate lives with the MLP page
        key, names = page
        if not self.cache.contains(key):
            return
        w = self.cache.get(key, names)
        p = f"model.layers.{nxt}"
        k = self.cfg.num_experts_per_tok
        ln = w.get(f"{p}.post_attention_layernorm.weight")
        h = mx.fast.rms_norm(x, ln, self.cfg.rms_norm_eps) if ln is not None else x
        parent = self.cfg.moe_module_prefix()
        moe = f"{p}.{parent}" if parent else p
        router_w = w.get(f"{moe}.router.weight")
        gate_w = w.get(f"{moe}.gate.weight")
        if router_w is not None:  # gpt-oss: linear router + bias, top-k on logits
            logits = h @ router_w.T
            bias = w.get(f"{moe}.router.bias")
            if bias is not None:
                logits = logits + bias
            idx = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        elif gate_w is not None:
            if self.cfg.model_type in ("glm_moe_dsa", "kimi_k25", "glm4_moe_lite"):
                scores = h.astype(mx.float32) @ gate_w.astype(mx.float32).T
            else:
                scores = (h @ gate_w.T).astype(mx.float32)
            bias = w.get(f"{moe}.gate.e_score_correction_bias")
            if bias is not None:  # GLM noaux_tc: SELECTION uses sigmoid + bias
                scores = mx.sigmoid(scores) + bias
            idx = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        else:  # dense layer (e.g. GLM first_k_dense_replace)
            return
        mx.eval(idx)
        for e in sorted({int(i) for i in idx.reshape(-1).tolist()}):
            self.prefetcher.schedule(
                self._expert_cache_key(nxt, e),
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
        if self.cfg.model_type == "deepseek_v4" and self.cfg.num_hash_layers:
            # Hash-routed layers need the token ids themselves, not just their
            # embeddings. Recorded here because this is the only point where
            # the ids and the sweep that consumes them are guaranteed to
            # correspond.
            self._dsv4_input_ids = mx.array(tokens)
        if self._embed_rows is not None:
            result = self._embed_rows.lookup(tokens)
        else:
            result = layer_runner.embed(mx.array(tokens), self._embed_weight())
        if self.cfg.model_type == "qwen4_exp":
            # PLE consumes the exact ids at its released decoder layer, while
            # the trunk carries four hyper-connection streams from embedding
            # through the final mixer.
            self._qwen4_input_ids = tuple(int(token) for token in tokens)
            result = mx.tile(result, (1, 1, self.cfg.qwen4_hc_count))
        if self.cfg.mup_enabled:
            # Afmoe (Trinity Nano/Mini): real modeling_afmoe.py applies this
            # ONCE right after embed_tokens, before any layer -- see
            # runtime/afmoe.py's module docstring.
            result = result * (self.cfg.hidden_size ** 0.5)
        return result

    def _lm_head_weight(self):
        if self.cfg.tie_word_embeddings:
            return (self._tied_lm_head_w if self._tied_lm_head_w is not None
                    else self._embed_weight())
        if self._streamed_lm_head is not None:
            return self._streamed_lm_head
        if self._lm_head_w is not None:
            return self._lm_head_w
        if (self._qwen4_lm_head_pin_suspended and bool(getattr(
                self.rc,
                "qwen4_serial_verify_suspend_lm_head",
                False))):
            # The verifier has consumed its last trunk page before asking for
            # logits.  Do not overlap that now-cold LRU tail with the 1.27-GB
            # exact head reload; this is a lifetime trim, not a cache-budget
            # mutation, so the next round may prefetch normally again.
            self._qwen4_serial_verify_head_restore_trim_bytes += int(
                self.cache.trim_to(0))
        head = self.cache.get(
            "lm_head", ["lm_head.weight"])["lm_head.weight"]
        if self._qwen35_lm_head_pin_suspended:
            self._restore_qwen35_serial_verify_lm_head(head)
        elif self._qwen4_lm_head_pin_suspended:
            self._restore_qwen4_phase_lm_head(head)
        return head

    def _suspend_qwen4_phase_lm_head(self) -> int:
        """Release the prior request's exact head before Qwen4 prefill."""
        if not bool(getattr(
                getattr(self, "rc", None), "qwen4_phase_lm_head", False)):
            return 0
        if self._lm_head_w is None:
            return 0
        started = time.perf_counter()
        self._lm_head_w = None
        released = self.cache.release_pinned(
            "qwen4:lm_head:persistent", ["lm_head.weight"])
        self._qwen4_lm_head_pin_suspended = bool(released)
        self._qwen4_phase_head_suspend_calls += 1
        self._qwen4_phase_head_suspend_bytes += int(released)
        self._qwen4_phase_head_suspend_s += time.perf_counter() - started
        return int(released)

    def _restore_qwen4_phase_lm_head(self, head=None) -> bool:
        """Promote the already-read post-prefill BF16 head without a copy."""
        if not self.rc.qwen4_phase_lm_head:
            return False
        started = time.perf_counter()
        self._qwen4_phase_head_restore_calls += 1
        promoted = self.cache.promote_to_pin(
            "lm_head", "qwen4:lm_head:persistent",
            tensors=(
                {"lm_head.weight": head}
                if head is not None else None
            ),
        )
        restored = promoted is not None
        if restored:
            self._lm_head_w = promoted["lm_head.weight"]
            self._qwen4_lm_head_pin_suspended = False
            self._qwen4_phase_head_restore_successes += 1
        else:
            self._qwen4_phase_head_restore_refusals += 1
        self._qwen4_phase_head_restore_s += time.perf_counter() - started
        return restored

    def _suspend_qwen4_serial_verify_lm_head(self) -> int:
        """Release the exact Qwen4 head for one target-body verifier sweep."""
        if not bool(getattr(
                getattr(self, "rc", None),
                "qwen4_serial_verify_suspend_lm_head",
                False)):
            return 0
        self._qwen4_serial_verify_head_suspend_calls += 1
        released = self._suspend_qwen4_phase_lm_head()
        self._qwen4_serial_verify_head_suspend_bytes += int(released)
        return int(released)

    def _suspend_qwen35_serial_verify_lm_head(self) -> int:
        """Drop the phase-scoped head before a streamed verifier trunk.

        The caller invokes this only after proposal projection has synchronized
        and before any target layer is fetched. Clearing ``_lm_head_w`` first
        ensures the cache owns the final live reference when it releases the
        dedicated pin page.
        """
        if (not bool(getattr(
                getattr(self, "rc", None),
                "qwen35_serial_verify_suspend_lm_head", False))
                or not bool(getattr(
                    self, "_qwen35_lm_head_suspend_request_active", False))):
            return 0
        if self._lm_head_w is None:
            return 0
        started = time.perf_counter()
        active_before = int(mx.get_active_memory())
        self._lm_head_w = None
        released = self.cache.release_pinned(
            "qwen35:lm_head:persistent", ["lm_head.weight"])
        active_after = int(mx.get_active_memory())
        active_released = max(0, active_before - active_after)
        self._qwen35_lm_head_pin_suspended = bool(released)
        self._qwen35_serial_verify_head_suspend_calls += 1
        self._qwen35_serial_verify_head_suspend_bytes += int(released)
        self._qwen35_serial_verify_head_suspend_active_released_bytes += (
            active_released)
        self._qwen35_serial_verify_head_suspend_active_peak_bytes = max(
            self._qwen35_serial_verify_head_suspend_active_peak_bytes,
            active_released,
        )
        self._qwen35_serial_verify_head_suspend_s += (
            time.perf_counter() - started)
        if self._qwen35_serial_verify_head_suspend_calls == 1:
            print(
                "[engine] Qwen phase head: "
                f"pin={int(released) / 1e6:.1f}MB, "
                f"active={active_before / 1e9:.3f}->"
                f"{active_after / 1e9:.3f}GB",
                flush=True,
            )
        return int(released)

    def _restore_qwen35_serial_verify_lm_head(self, head=None) -> bool:
        """Re-pin the verifier's existing demand head without another read."""
        if not self.rc.qwen35_serial_verify_suspend_lm_head:
            return False
        started = time.perf_counter()
        self._qwen35_serial_verify_head_restore_calls += 1
        promoted = self.cache.promote_to_pin(
            "lm_head", "qwen35:lm_head:persistent",
            tensors=(
                {"lm_head.weight": head}
                if head is not None else None
            ),
        )
        restored = promoted is not None
        if restored:
            self._lm_head_w = promoted["lm_head.weight"]
            self._qwen35_lm_head_pin_suspended = False
            self._qwen35_serial_verify_head_restore_successes += 1
        else:
            # A tight governor budget can make the verifier head pass-through.
            # Retain ordinary demand semantics instead of forcing a second
            # read or weakening the active memory limit.
            self._qwen35_serial_verify_head_restore_refusals += 1
        self._qwen35_serial_verify_head_restore_s += (
            time.perf_counter() - started)
        return restored

    def _final_logits(self, hidden: mx.array, head=None) -> mx.array:
        head = self._lm_head_weight() if head is None else head
        if self.cfg.model_type == "qwen4_exp":
            from .qwen4_exp import final_logits

            return final_logits(
                hidden, self._qwen4_final_mixer_w, head, self.cfg)
        if self.cfg.model_type in ("qwen3_5_moe", "qwen3_5"):
            from .qwen35 import final_logits

            return final_logits(
                hidden, self._norm_w, head, self.cfg.rms_norm_eps)
        return layer_runner.final_logits(
            hidden, self._norm_w, head, self.cfg.rms_norm_eps)

    def _constraint_logits(
        self, logits: mx.array, constraint, hidden: mx.array | None = None,
    ) -> mx.array:
        """Apply a grammar before sparse-head candidate selection.

        Ordinary dense/streamed heads retain their established post-projection
        mask. A candidate-reranked head instead selects its exact BF16 rows
        from the legal approximate logits so every emitted candidate remains
        grammar-valid. The operation depends only on the constraint state, not
        prompt wording, subject, or tool identity.
        """
        if constraint is None:
            return logits
        from .quant import (
            RerankedQHead,
            reranked_lm_head_capture_scope,
            reranked_matmul,
        )

        head = self._lm_head_weight()
        source = hidden if hidden is not None else self._h_last
        if isinstance(head, RerankedQHead) and source is not None:
            if self.cfg.model_type == "qwen4_exp":
                from .qwen4_exp import final_hidden

                normalized = final_hidden(
                    source[:, -1:, :],
                    self._qwen4_final_mixer_w,
                    self.cfg,
                )
            elif self.cfg.model_type in ("qwen3_5_moe", "qwen3_5"):
                from .qwen35 import qwen35_rms_norm

                normalized = qwen35_rms_norm(
                    source[:, -1:, :], self._norm_w,
                    self.cfg.rms_norm_eps)
            else:
                normalized = mx.fast.rms_norm(
                    source[:, -1:, :], self._norm_w,
                    self.cfg.rms_norm_eps)
            # A constrained request marks its ordinary unrestricted head
            # projections provisional at the server request boundary. Only
            # this grammar-aware rerank is an authoritative target decision
            # and therefore eligible for promotion evidence.
            with reranked_lm_head_capture_scope(
                    head, "authoritative-target"):
                return reranked_matmul(
                    normalized, head,
                    logits_transform=constraint.mask_logits)[0, 0]
        return constraint.mask_logits(logits)

    def _all_logits(self, hidden: mx.array) -> mx.array:
        head = self._lm_head_weight()
        if self.cfg.model_type == "qwen4_exp":
            from .qwen4_exp import all_logits

            return all_logits(
                hidden, self._qwen4_final_mixer_w, head, self.cfg)
        if self.cfg.model_type in ("qwen3_5_moe", "qwen3_5"):
            from .qwen35 import all_logits

            return all_logits(
                hidden, self._norm_w, head, self.cfg.rms_norm_eps)
        return layer_runner.all_logits(
            hidden, self._norm_w, head, self.cfg.rms_norm_eps)

    def begin_dspark_tap_capture(self, collector) -> None:
        if self._dspark_tap_collector is not None:
            raise RuntimeError("a DSpark target-tap capture is already active")
        expected = tuple(int(v) for v in collector.tap_layers)
        if not expected:
            raise ValueError("DSpark target-tap capture needs at least one layer")
        if expected != tuple(sorted(set(expected))):
            raise ValueError("DSpark target-tap layers must be unique and ordered")
        if expected[-1] >= self.cfg.num_hidden_layers:
            raise ValueError("DSpark target-tap layer is outside the target")
        self._dspark_tap_collector = collector

    def end_dspark_tap_capture(self, collector) -> None:
        if self._dspark_tap_collector is not collector:
            raise RuntimeError("DSpark target-tap capture ownership changed")
        self._dspark_tap_collector = None

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
        collector = (
            self._dspark_tap_collector if tap_layers is None else None)
        if collector is not None:
            tap_layers = collector.tap_layers
        self._tap_hidden = {}
        position_count = int(x.shape[1])
        profiler = self._request_profiler
        # Never charge one-position decode for a multi-position prefill
        # high-water, or a smaller retry chunk for a larger prefill's
        # high-water.  The first layer of each exact position-count class
        # measures its own scratch, after which subsequent layers reserve that
        # class-specific maximum.  A single global maximum made a live
        # Qwen3.6-35B-A3B run finish prefill, then reject decode because it
        # inherited 1.28 GB of prefill scratch that decode never allocates;
        # a two-class version then made 8-token retries inherit the 32-token
        # prefill maximum.  Indexing by actual tensor width fixes both.
        (self._layer_transient,
         self._layer_transient_margin) = _layer_transient_for_positions(
             position_count,
             getattr(
                 self, "_prefill_layer_transient_by_positions", {}
             ).get(position_count, 0),
             getattr(self, "_decode_layer_transient", 0))
        n = self.cfg.num_hidden_layers
        moe = bool(self.cfg.num_experts)
        if self._resident_moe_layers is not None and tap_layers is None:
            if profiler is not None:
                profiler.begin_sweep(
                    position_count, path="resident_moe_stack")
                stack_t0 = time.perf_counter()
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
            if profiler is not None:
                mx.eval(x)
                profiler.record_stack(
                    positions=position_count, path="resident_moe_stack",
                    wall_s=time.perf_counter() - stack_t0)
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
            self.rc.resident_fast_decode
            # Speculative verify/refeed sweeps are decode-shaped work at
            # small widths (catchup + forced prefix + <=k proposals), not
            # prefill: without this, every verify sweep pays the ordinary
            # loop's per-layer sync cost while plain decode enjoys the
            # fast path, inverting speculation's economics. The hint is set
            # only by speculative wrappers around their own forward_tokens
            # calls (see qwen35_ngram.py), never for real prefill.
            and (x.shape[1] == 1
                 or (x.shape[1] <= 48
                     and getattr(self, "_speculative_verify_hint", False)))
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
            if profiler is not None:
                profiler.begin_sweep(
                    position_count, path="resident_fast_stack")
                stack_t0 = time.perf_counter()
            if self.cfg.model_type == "qwen3_5":
                # 2026-07-23: hybrid resident fast decode. Two things at
                # once: (1) the unconditional run_block below was a latent
                # crash for qwen3_5 (a plain dense block that looks up
                # self_attn.* names linear_attention layers don't have) --
                # the path simply never fired for qwen3_5 before because
                # server.py never set resident_fast_decode for it; (2) the
                # ordinary per-layer loop costs ~56 GPU sync points per
                # decode token for this hybrid (32 per-layer mx.eval(x) + 24
                # per-DeltaNet-layer mx.eval(state)), which dominates the
                # resident compute-bound regime. Run every layer lazily
                # (defer_state_eval) and batch-eval all updated recurrent
                # state in ONE call at the sweep boundary -- identical
                # arithmetic, one sync instead of ~56.
                from .qwen35 import run_qwen35_block

                for i, w in enumerate(fast_layers):
                    x = run_qwen35_block(
                        x, w, f"model.layers.{i}", self.cfg, kv, i, offset,
                        self._get_experts,
                        iter_expert_batches=self._iter_expert_batches,
                        zmlx_fused_decode=self.rc.zmlx_fused_deltanet_decode,
                        native_fused_decode=self.rc.native_fused_deltanet_decode,
                        chunked_delta_prefill=(
                            self.rc.qwen_chunked_delta_prefill),
                        compiled_delta_prefill=(
                            self.rc.qwen_compiled_delta_prefill),
                        native_fused_delta_prefill=(
                            self.rc.qwen_native_fused_delta_prefill),
                        defer_state_eval=True,
                    )
                kda = getattr(kv, "kda_cache", None)
                if kda is not None:
                    pending_state = [
                        s for s in kda._state if s is not None]
                    pending_state.extend(
                        value
                        for history in kda._conv if history is not None
                        for value in history if value is not None)
                    if pending_state:
                        mx.eval(*pending_state)
                if profiler is not None and profiler.sync_substeps:
                    mx.eval(x)
                    profiler.record_stack(
                        positions=position_count,
                        path="resident_fast_stack",
                        wall_s=time.perf_counter() - stack_t0)
                return x
            for i, w in enumerate(fast_layers):
                last_only = (
                    final_mlp_last_only and i == n - 1 and x.shape[1] > 1)
                x = layer_runner.run_block(
                    x, w, f"model.layers.{i}", self.cfg, kv, i, offset,
                    mlp_last_only=last_only,
                    rope_freqs=self._rope_freqs, rope_mscale=self._mscale,
                    fused_swiglu=self.rc.fused_swiglu,
                )
            if profiler is not None and profiler.sync_substeps:
                mx.eval(x)
                profiler.record_stack(
                    positions=position_count, path="resident_fast_stack",
                    wall_s=time.perf_counter() - stack_t0)
            return x
        if profiler is not None:
            profiler.begin_sweep(position_count, path="streamed")
        # F128: Kimi K3's real AttnRes mechanism needs one extra piece of
        # state (block_residual) threaded through every layer of THIS sweep
        # only -- the real reference re-inits it fresh at the top of every
        # forward() call and never persists it across calls, so a fresh
        # empty block_residual here (one per _sweep call = one per chunk,
        # exactly matching a chunk-major caller's own "one forward-call per
        # chunk" mapping) is correct; see run_kimi_k3_block's own docstring.
        # F213: DeepSeek V4 carries the hidden state as hc_mult parallel
        # streams for the whole depth. The released Transformer.forward
        # expands right after the embedding with
        # ``h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`` and reduces once via
        # hc_head after the last layer, so this carrier is per-sweep exactly
        # like block_residual below.
        hc_stream = None
        if self.cfg.model_type == "deepseek_v4":
            # One position per sweep, shared by every layer. A real prefill
            # (width > 1 at offset 0) resets it; each later sweep advances by
            # the width it actually consumed.
            if offset == 0 and x.shape[1] > 1:
                kv.dsv4_pos = 0
            resolved = int(getattr(kv, "dsv4_pos", 0)) if offset == 0 else offset
            kv.dsv4_sweep_pos = resolved
            kv.dsv4_pos = resolved + x.shape[1]
            hc_stream = mx.broadcast_to(
                x[:, :, None, :],
                (x.shape[0], x.shape[1], self.cfg.hc_mult, x.shape[2]))
        elif self.cfg.model_type == "glm5_next":
            # GLM-5.3 starts every forward over the current positions by
            # expanding embeddings into four identical mHC streams, then
            # reduces them with an unweighted mean after the final block.
            hc_stream = mx.broadcast_to(
                x[:, :, None, :],
                (x.shape[0], x.shape[1], self.cfg.hc_mult, x.shape[2]))
        block_residual = (
            (
                []
                if self.rc.kimi_k3_fused_attnres_tile_size
                else mx.zeros(
                    (x.shape[0] * x.shape[1], 0, x.shape[2]),
                    dtype=x.dtype,
                )
            )
            if self.cfg.model_type == "kimi_k3" else None)
        for i in range(n):
            self._select_layer_transient(position_count, i)
            # F36: on the last layer of a prefill whose only consumer is the last
            # position's logits, MLP outputs for earlier positions are dead —
            # attention still runs full-width so the KV cache stays complete.
            last_only = final_mlp_last_only and i == n - 1 and x.shape[1] > 1
            if self.prefetcher:
                for j in range(i + 1, min(i + 1 + self.rc.prefetch_depth, n)):
                    hint = self._layer_fetch_bytes_estimate(j)
                    self.prefetcher.schedule(
                        self._layer_key(j),
                        self._layer_names(j),
                        page_size_hint=hint or None,
                    )

            cache_before = (
                profiler.cache_snapshot(self.cache)
                if profiler is not None else None)
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
            if self._dsv4_packed_trunk:
                w = self._materialize_packed_trunk(w)
            weight_wait_s = time.perf_counter() - t0
            self.timer.add("weights_wait", weight_wait_s)

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
                self.governor.reserve(
                    self._layer_transient,
                    margin=self._layer_transient_margin)

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
                    iter_expert_batches=(
                        self._iter_expert_batches
                        if (self.rc.expert_fetch_batch
                            or self.rc.expert_batch_prefetch)
                        else None),
                    profile=profiler,
                )
            elif self.cfg.model_type in ("glm_moe_dsa", "kimi_k25", "glm4_moe_lite"):
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
                    profile=profiler,
                )
            elif self.cfg.model_type == "deepseek_v4":
                from .deepseek_v4 import run_deepseek_v4_block

                prefix = f"model.layers.{i}"
                hc_stream = run_deepseek_v4_block(
                    hc_stream,
                    {key: w[f"{prefix}.hc_{key}"] for key in (
                        "attn_fn", "attn_scale", "attn_base",
                        "ffn_fn", "ffn_scale", "ffn_base")},
                    {"attn": w[f"{prefix}.attn_norm.weight"],
                     "ffn": w[f"{prefix}.ffn_norm.weight"]},
                    lambda t: self._deepseek_v4_attention(t, w, prefix, i, kv,
                                                          offset),
                    lambda t: self._deepseek_v4_ffn(t, w, prefix, i),
                    hc_mult=self.cfg.hc_mult,
                    norm_eps=self.cfg.rms_norm_eps,
                    sinkhorn_iters=self.cfg.hc_sinkhorn_iters,
                    hc_eps=self.cfg.hc_eps)
                if self._dspark_capture is not None and i in self._dspark_targets:
                    # The released target appends h.mean(dim=2) -- the mean over
                    # the hyper-connection streams, NOT the hc_pre reduction --
                    # after each target layer, in dspark_target_layer_ids order.
                    self._dspark_capture.append(mx.mean(hc_stream, axis=2))
                # Materialize per layer. The gathered sparse-attention operand
                # is [1, positions, window + compressed, head_dim] -- about
                # 124MB at a 300-position prompt -- and left lazy every
                # layer's copy stays live, reaching >12GB across 43 layers
                # while each individual reservation looked small.
                mx.eval(hc_stream)
            elif self.cfg.model_type == "glm5_next":
                from .glm5_next import run_glm5_next_block

                hc_stream = run_glm5_next_block(
                    hc_stream, w, f"model.layers.{i}", self.cfg, kv, i,
                    offset, self._get_experts,
                    mlp_last_only=last_only,
                    iter_expert_batches=self._iter_expert_batches,
                    native_fused_kda_decode=(
                        self.rc.native_fused_deltanet_decode),
                    native_fused_kda_prefill=(
                        self.rc.glm53_native_fused_kda_prefill),
                    compiled_kda_prefill=(
                        self.rc.glm53_compiled_kda_prefill),
                    compiled_kda_prefill_segment=(
                        self.rc.glm53_compiled_kda_segment),
                    profile=profiler,
                )
                mx.eval(hc_stream)
            elif self.cfg.model_type == "lfm2":
                # F202: 22 gated short-conv + 8 full-attention layers. The conv
                # layers keep a fixed conv_L_cache-1 history in the same
                # KDAStateCache companion slot Kimi's causal convolution uses,
                # so fork/restore, disk spill, and suffix-decoding rollback all
                # apply unchanged. No experts: LFM2 is dense.
                from .lfm2 import run_lfm2_block

                x = run_lfm2_block(
                    x, w, f"model.layers.{i}", self.cfg, kv, i, offset,
                    state_cache=getattr(kv, "kda_cache", None),
                    mlp_last_only=last_only,
                    profile=profiler,
                )
            elif self.cfg.model_type == "kimi_linear":
                # kimi_k3 dispatches separately below -- run_kimi_linear_block
                # has no AttnRes awareness, which kimi_k3's real checkpoint
                # (attn_res_block_size=12, confirmed active) genuinely needs.
                from .kimi_linear import run_kimi_linear_block

                x = run_kimi_linear_block(
                    x, w, f"model.layers.{i}", self.cfg, kv, i, offset, self._get_experts,
                    mlp_last_only=last_only,
                    iter_expert_batches=self._iter_expert_batches,
                    native_fused_decode=self.rc.native_fused_deltanet_decode,
                    profile=profiler,
                )
            elif self.cfg.model_type == "kimi_k3":
                # F128: AttnRes-aware block runner -- see run_kimi_k3_block's
                # docstring. block_residual (initialized above, before this
                # loop) is threaded layer-to-layer exactly like x itself;
                # mlp_last_only is honored only AFTER the full layer
                # (attention+MLP) completes, not between them like
                # run_kimi_linear_block's version, so block_residual's
                # row count (batch*positions) never mismatches x's across
                # this loop -- see run_kimi_k3_block's own docstring for why.
                from .kimi_linear import run_kimi_k3_block

                x, block_residual = run_kimi_k3_block(
                    x, w, f"model.layers.{i}", self.cfg, kv, i, offset,
                    block_residual, self._get_experts,
                    mlp_last_only=last_only,
                    iter_expert_batches=self._iter_expert_batches,
                    native_fused_decode=self.rc.native_fused_deltanet_decode,
                    native_fused_prefill=(
                        self.rc.kimi_k3_native_fused_kda_prefill),
                    compiled_prefill=(
                        self.rc.kimi_k3_compiled_kda_prefill),
                    profile=profiler,
                    fused_attnres_tile_size=(
                        self.rc.kimi_k3_fused_attnres_tile_size),
                )
            elif self.cfg.model_type == "qwen4_exp":
                from .qwen4_exp import run_qwen4_block

                input_ids = getattr(self, "_qwen4_input_ids", ())
                if len(input_ids) != int(x.shape[1]):
                    raise ValueError(
                        "Qwen4 sweep lost its position-aligned input ids")
                x = run_qwen4_block(
                    x, input_ids, w, f"model.layers.{i}", self.cfg,
                    kv, i, offset, self._get_experts,
                    row_store=self._qwen4_ple_rows,
                    iter_expert_batches=self._iter_expert_batches,
                    profile=profiler,
                    compiled_delta_prefill=(
                        self.rc.qwen_compiled_delta_prefill),
                    native_fused_delta_prefill=(
                        self.rc.qwen_native_fused_delta_prefill),
                )
                if last_only:
                    x = x[:, -1:, :]
                    self._qwen4_input_ids = (input_ids[-1],)
            elif self.cfg.model_type in ("qwen3_5_moe", "qwen3_5"):
                from .qwen35 import run_qwen35_block

                x = run_qwen35_block(
                    x, w, f"model.layers.{i}", self.cfg, kv, i, offset,
                    self._get_experts, mlp_last_only=last_only,
                    iter_expert_batches=self._iter_expert_batches,
                    zmlx_fused_decode=self.rc.zmlx_fused_deltanet_decode,
                    native_fused_decode=self.rc.native_fused_deltanet_decode,
                    chunked_delta_prefill=(
                        self.rc.qwen_chunked_delta_prefill),
                    compiled_delta_prefill=(
                        self.rc.qwen_compiled_delta_prefill),
                    native_fused_delta_prefill=(
                        self.rc.qwen_native_fused_delta_prefill),
                    profile=profiler,
                )
            elif self.cfg.model_type == "jet_nemotron":
                from .jet_nemotron import run_jet_nemotron_block

                x = run_jet_nemotron_block(
                    x, w, f"model.layers.{i}", self.cfg, kv, i, offset,
                    rope_freqs=self._rope_freqs, mlp_last_only=last_only,
                    native_fused_decode=self.rc.native_fused_deltanet_decode,
                )
            elif self.cfg.model_type == "afmoe":
                from .afmoe import run_afmoe_block

                x = run_afmoe_block(
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
            end_active = mx.get_active_memory()
            peak_active = mx.get_peak_memory()
            measured_transient = _resident_adjusted_transient(
                active_before, end_active, peak_active)
            if self.cfg.model_type == "kimi_k3":
                self._last_k3_transient_observation = {
                    "layer": i,
                    "signature": self._transient_layer_signature(i),
                    "start_active_bytes": int(active_before),
                    "end_active_bytes": int(end_active),
                    "peak_active_bytes": int(peak_active),
                    "measured_transient_bytes": int(measured_transient),
                }
            self._record_layer_transient(
                position_count, i, measured_transient)
            self._note_true_peak()
            if position_count == 1 and self.cfg.model_type in (
                "kimi_k3", "glm_moe_dsa"
            ):
                # A token's layer-i endpoint is consumed only by layer i of
                # the *next* token. No later layer in this sweep reads it, so
                # return exact KDA/MLA state to the configured spill tier as
                # soon as its compute and peak accounting are complete. This
                # is the decode analogue of the proven layer-stationary
                # prefill lifetime and prevents a restored 27K endpoint from
                # accumulating every reloaded layer in Metal at once.
                spilled = False
                if (
                    self.rc.kimi_k3_kda_spill_dir
                    and self.cfg.model_type == "kimi_k3"
                    and i in self.cfg.kda_layers
                    and getattr(kv, "kda_cache", None) is not None
                ):
                    spilled = kv.kda_cache.spill_layer(i) or spilled
                if (
                    self.rc.kimi_k3_mla_kv_spill_dir
                    and self.cfg.model_type == "kimi_k3"
                    and i in self.cfg.full_attn_layers
                    and getattr(kv, "latent_spill_enabled", False)
                ):
                    spilled = kv.spill_latent_layer(i) or spilled
                if (
                    self.rc.glm_dsa_mla_kv_spill_dir
                    and self.cfg.model_type == "glm_moe_dsa"
                    and getattr(kv, "latent_spill_enabled", False)
                ):
                    spilled = kv.spill_latent_layer(i) or spilled
                if spilled:
                    mx.clear_cache()
            compute_s = time.perf_counter() - t0
            self.timer.add("layer_compute", compute_s)
            if profiler is not None:
                profiler.record_layer(
                    i, positions=position_count,
                    weight_wait_s=weight_wait_s, compute_s=compute_s,
                    cache_before=cache_before,
                    cache_after=profiler.cache_snapshot(self.cache),
                    layer_type=self._profile_layer_type(i),
                )
            if tap_layers is not None and i in tap_layers:
                tap_hidden = (
                    mx.mean(hc_stream, axis=2)
                    if self.cfg.model_type == "glm5_next" else x)
                self._tap_hidden[i] = tap_hidden
                if collector is not None:
                    collector.observe(i, tap_hidden, position_start=offset)
            if (self.rc.router_lookahead and moe and self.prefetcher
                    and i + 1 < n and x.shape[1] == 1):
                # F45 — decode only: prefill's multi-position unions flooded the
                # cache and halved hit rates (measured; see benchmark_results)
                self._router_lookahead(x, i + 1)
            del w
        self._restore_aggregate_layer_transient(position_count)
        if self.cfg.model_type == "deepseek_v4" and hc_stream is not None:
            # One reduction after the last layer, matching the released
            # forward: hc_head consumes the whole mix vector as sigmoid gates
            # and needs no Sinkhorn. The caller applies model.norm and the LM
            # head right after _sweep returns.
            from .deepseek_v4 import hc_head

            x = hc_head(
                hc_stream, self._hc_head_fn, self._hc_head_scale,
                self._hc_head_base, norm_eps=self.cfg.rms_norm_eps,
                eps=self.cfg.hc_eps)
            if final_mlp_last_only and x.shape[1] > 1:
                x = x[:, -1:, :]
            return x
        if self.cfg.model_type == "glm5_next" and hc_stream is not None:
            # The released Glm5NextTextHyperHead has no learned parameters:
            # it is exactly the mean over four streams before model.norm.
            x = mx.mean(hc_stream, axis=2).astype(hc_stream.dtype)
            if final_mlp_last_only and x.shape[1] > 1:
                x = x[:, -1:, :]
            return x
        if self.cfg.model_type == "kimi_k3" and block_residual is not None:
            # F128: the real KimiLinearModel.forward applies this ONCE,
            # after every layer, before its own final model.norm -- which
            # this function's caller (_final_logits/_all_logits) applies
            # right after _sweep returns. run_kimi_k3_block deliberately
            # never trims x mid-loop (see its own docstring) so x and
            # block_residual stay row-aligned through this call; the
            # final_mlp_last_only trim vOOM's OTHER model types apply mid-
            # loop happens here instead, after the real readout, once.
            from .kimi_linear import apply_output_attn_res

            x = apply_output_attn_res(
                x, {
                    "model.output_attn_res_proj.weight": self._output_attn_res_proj_w,
                    "model.output_attn_res_norm.weight": self._output_attn_res_norm_w,
                }, block_residual, self.cfg,
                fused_tile_size=(
                    self.rc.kimi_k3_fused_attnres_tile_size))
            if final_mlp_last_only and x.shape[1] > 1:
                x = x[:, -1:, :]
        return x

    def _layer_stationary_qwen35_sweep(
            self, x: mx.array, kv: KVCache, offset: int,
            tile_width: int, on_progress=None, *, layer_start: int = 0,
            layer_end: int | None = None,
            profile_path: str = "layer_stationary_qwen35",
            positions3: mx.array | None = None,
            boundary_fork_at: int | None = None,
            boundary_fork_kv: KVCache | None = None) -> mx.array:
        """F94 live path: layer-major (not chunk-major) prefill for qwen3_5/
        qwen3_5_moe (Qwen3.5-4B/9B, Qwen3.6-27B hybrid DeltaNet/full-attention
        layers). Fetches each layer's weights exactly once for the WHOLE
        prefill range in `x`, unlike `_sweep` (called once per chunk by
        generate()'s prefill loop), which re-fetches every layer's weights
        once per chunk whenever the resident cache can't hold the whole
        model. See CLAUDE.md's 2026-07-23 correction: measured single-layer
        fetches already run at raw disk bandwidth (~1.6 GB/s) -- the real
        cost is fetching the SAME layers repeatedly across chunks, not any
        per-fetch overhead. Deliberately narrower than layer_stationary.py
        (which is model-agnostic but untested against real recurrent state).

        F94 (2026-07-20) originally covered only the dense (num_experts=0)
        case by dispatching straight to run_qwen35_block per layer per tile.
        **Extended 2026-07-25** to qwen3_5_moe (Qwen3.6-27B/35B-A3B,
        Qwen3.5-35B-A3B's routed layers) the same way F35 extended Kimi
        Linear/GLM: attention still runs per TILE via
        `_qwen35_attention_residual` (DeltaNet state and ordinary KV both
        still need causal tile order, unchanged), but
        `_qwen35_mlp_residual` (MoE routing + expert fetch, when
        cfg.num_experts>0) now runs exactly ONCE per layer on the full
        tile-concatenated attention output, instead of once per tile inside
        the old single run_qwen35_block call -- eliminating the same
        cross-chunk redundant expert re-routing/re-fetching F35 fixed for
        Kimi Linear/GLM. For the dense case (num_experts=0) this is a no-op
        change: running _swiglu once over the concatenated range is the same
        deterministic per-position function as running it once per tile,
        since dense MLP has no cross-position state -- so this generalizes
        the sweep without altering dense qwen3_5's own already-proven
        behavior.

        Mirrors _sweep's per-layer governor reservation, transient-memory
        tracking, and prefetch scheduling exactly (just reordered: tiles of
        the whole prompt nested inside the layer loop, rather than one
        chunk's positions swept across all layers) so it inherits the same
        memory-safety behavior _sweep already has, not a weaker or different
        one. KDAStateCache correctness across tile boundaries follows from
        each layer's own recurrent state depending only on that layer's own
        sequential inputs and its own prior state -- never on another layer's
        loop position -- so re-associating the (layer, tile) loop nesting
        cannot change any layer's own state evolution; MoE routing is
        stateless per call (a function of `h` alone), so computing it once
        over all positions is the same function evaluated on the union of
        its arguments, not a different function -- the same argument F35
        already used for Kimi Linear/GLM. Proven directly (not just argued)
        in tests/test_f94_qwen35_layer_stationary_oracle.py (dense) and
        tests/test_f35_qwen35_moe_layer_stationary_oracle.py (MoE).
        """
        from .qwen35 import _qwen35_attention_residual, _qwen35_mlp_residual

        if tile_width <= 0:
            raise ValueError("tile_width must be positive")
        n = self.cfg.num_hidden_layers
        layer_end = n if layer_end is None else int(layer_end)
        layer_start = int(layer_start)
        if not 0 <= layer_start < layer_end <= n:
            raise ValueError(
                "Qwen layer-stationary range must satisfy "
                "0 <= layer_start < layer_end <= num_hidden_layers")
        total = int(x.shape[1])
        if positions3 is not None and tuple(positions3.shape) != (3, total):
            raise ValueError(
                "Qwen layer-stationary positions must have shape "
                f"(3, {total})")
        if boundary_fork_at is not None:
            boundary_fork_at = int(boundary_fork_at)
            if not 0 < boundary_fork_at < total:
                raise ValueError(
                    "Qwen boundary fork must be strictly inside the sweep")
            if type(boundary_fork_kv) is not KVCache:
                raise TypeError(
                    "Qwen layer-stationary boundary fork requires plain KVCache")
            if getattr(boundary_fork_kv, "kda_cache", None) is None:
                raise ValueError(
                    "Qwen layer-stationary boundary fork is missing recurrent state")
        elif boundary_fork_kv is not None:
            raise ValueError("boundary_fork_kv requires boundary_fork_at")
        dense_mlp = not bool(self.cfg.num_experts)
        # A dense MLP is position-independent.  Keep it on the same bounded
        # row tiles as attention instead of materializing gate/up/down
        # activations for the entire prompt (49K rows is ~11 GB for the 27B
        # checkpoint).  Besides bounding memory, this matches the ordinary
        # chunk-major dense path's call shapes.  Routed MoE retains its
        # existing whole-range union so expert batching/fetch behavior is
        # unchanged.
        # Tile width controls operator scratch, but layer-stationary execution
        # also retains/concatenates one output row per sequence position. A
        # 30K shallow sweep and a 1K deep suffix with the same 32-row tile are
        # therefore different memory shapes and must not share one learned
        # high-water key.
        transient_shape_positions = total
        (self._layer_transient,
         self._layer_transient_margin) = _layer_transient_for_positions(
             transient_shape_positions,
             getattr(
                 self, "_prefill_layer_transient_by_positions", {}
             ).get(transient_shape_positions, 0),
             getattr(self, "_decode_layer_transient", 0))
        profiler = self._request_profiler
        tap_collector = self._dspark_tap_collector
        if profiler is not None:
            profiler.begin_sweep(total, path=profile_path)
        for i in range(layer_start, layer_end):
            self._select_layer_transient(transient_shape_positions, i)
            if self.prefetcher:
                for j in range(
                        i + 1,
                        min(i + 1 + self.rc.prefetch_depth, layer_end)):
                    self.prefetcher.schedule(self._layer_key(j), self._layer_names(j))

            cache_before = (
                profiler.cache_snapshot(self.cache)
                if profiler is not None else None)
            t0 = time.perf_counter()
            layer_key = self._layer_key(i)
            layer_names = self._layer_names(i)
            if not self.cache.contains(layer_key):
                incoming_page = self._layer_fetch_bytes_estimate(i)
                if incoming_page:
                    self.cache.prepare_for(incoming_page)
                    if self.governor is not None:
                        self.governor.reserve(
                            incoming_page,
                            reason="qwen-prefill-layer-page",
                        )
            w = self.cache.get(layer_key, layer_names)
            if self._dsv4_packed_trunk:
                w = self._materialize_packed_trunk(w)
            weight_wait_s = time.perf_counter() - t0
            self.timer.add("weights_wait", weight_wait_s)

            active_before = mx.get_active_memory()
            mx.reset_peak_memory()
            tiles = []
            mlp_tile = None
            x_after_attn = None
            pos = 0
            t0 = time.perf_counter()
            dense_mlp_s = 0.0
            while pos < total:
                end = min(pos + tile_width, total)
                # A stable chat boundary can fall inside an ordinary prefill
                # tile. Land on it exactly so this layer's recurrent/KV
                # endpoint can be retained before the generation scaffold is
                # consumed, while the same resident weights immediately
                # continue through that scaffold. No arithmetic is reordered
                # inside either side of the boundary.
                if (boundary_fork_at is not None
                        and pos < boundary_fork_at < end):
                    end = boundary_fork_at
                # Same per-tile proactive reserve _sweep does once per chunk
                # (matching that granularity, not just once per layer), so a
                # real mid-sweep pressure spike is caught at the same
                # resolution the chunk-major path already catches it at.
                if self.governor is not None and self._layer_transient:
                    signature = self._transient_layer_signature(i)
                    observations = int(getattr(
                        self, "_layer_transient_observation_counts", {}
                    ).get((transient_shape_positions, signature), 0))
                    completed_output_bytes = (
                        pos * int(x.nbytes) // total if dense_mlp else 0)
                    scratch_reserve = _remaining_layer_transient_reserve(
                        self._layer_transient, completed_output_bytes)
                    reserve_margin = (
                        _recurring_layer_transient_reserve_margin(
                            transient_shape_positions, observations))
                    try:
                        self.governor.reserve(
                            scratch_reserve,
                            margin=reserve_margin,
                            reason="qwen-prefill-transient",
                        )
                    except MemoryError:
                        print(
                            "[qwen35-prefill-admission] "
                            f"layer={i} "
                            f"signature={signature} "
                            f"tile={end - pos} position={pos}/{total} "
                            f"active={int(mx.get_active_memory())} "
                            f"scratch={int(scratch_reserve)} "
                            f"measured_scratch={int(self._layer_transient)} "
                            f"completed_output={completed_output_bytes} "
                            f"observations={observations} "
                            f"margin={int(reserve_margin)}",
                            flush=True,
                        )
                        raise
                xt = x[:, pos:end, :]
                tile_positions3 = (
                    None if positions3 is None else positions3[:, pos:end])
                attention_t0 = time.perf_counter()
                yt = _qwen35_attention_residual(
                    xt, w, f"model.layers.{i}", self.cfg, kv, i,
                    offset + pos, mlp_last_only=False,
                    positions3=tile_positions3,
                    zmlx_fused_decode=self.rc.zmlx_fused_deltanet_decode,
                    native_fused_decode=self.rc.native_fused_deltanet_decode,
                    chunked_delta_prefill=(
                        self.rc.qwen_chunked_delta_prefill),
                    compiled_delta_prefill=(
                        self.rc.qwen_compiled_delta_prefill),
                    native_fused_delta_prefill=(
                        self.rc.qwen_native_fused_delta_prefill),
                )
                mx.eval(yt)
                if (boundary_fork_at is not None
                        and end == boundary_fork_at):
                    # KVCache/KDAStateCache updates replace arrays rather than
                    # mutating them in place. Sharing the evaluated endpoint
                    # here is therefore copy-on-write: the following scaffold
                    # tile installs new arrays in ``kv`` without changing the
                    # retained stable-boundary branch.
                    fork_arrays = []
                    keys = kv.keys[i]
                    values = kv.values[i]
                    if keys is not None:
                        fork_arrays.extend((keys, values))
                    recurrent = getattr(kv, "kda_cache", None)
                    fork_recurrent = boundary_fork_kv.kda_cache
                    state = recurrent.state(i) if recurrent is not None else None
                    history = (
                        recurrent.conv_history(i)
                        if recurrent is not None else None)
                    if state is not None:
                        fork_arrays.append(state)
                    if history is not None:
                        fork_arrays.extend(
                            value for value in history if value is not None)
                    if fork_arrays:
                        mx.eval(*fork_arrays)
                    if keys is not None:
                        boundary_fork_kv.keys[i] = keys
                        boundary_fork_kv.values[i] = values
                        boundary_fork_kv._windows[i] = kv._windows[i]
                        boundary_fork_kv._starts[i] = kv._starts[i]
                    if state is not None:
                        fork_recurrent.set_state(i, state)
                    if history is not None:
                        fork_recurrent.set_conv_history(i, tuple(history))
                if profiler is not None and profiler.sync_substeps:
                    profiler.record_substep(
                        "attention", i,
                        time.perf_counter() - attention_t0,
                        positions=end - pos)
                if dense_mlp:
                    # Dense SwiGLU is independent at every position and does
                    # not participate in the next tile's attention/KV update.
                    # Finish this tile now so the attention residual can be
                    # released instead of retaining both prompt-sized
                    # attention and MLP tile lists until the layer ends.
                    tile_mlp_t0 = time.perf_counter()
                    mlp_tile = _qwen35_mlp_residual(
                        yt, w, f"model.layers.{i}", self.cfg, i,
                        self._get_experts,
                        iter_expert_batches=self._iter_expert_batches,
                        profile=profiler)
                    mx.eval(mlp_tile)
                    dense_mlp_s += time.perf_counter() - tile_mlp_t0
                    tiles.append(mlp_tile)
                    yt = None
                    mlp_tile = None
                else:
                    tiles.append(yt)
                pos = end
            mlp_t0 = time.perf_counter()
            if dense_mlp:
                x = (
                    tiles[0]
                    if len(tiles) == 1
                    else mx.concatenate(tiles, axis=1))
            else:
                x_after_attn = (
                    tiles[0]
                    if len(tiles) == 1
                    else mx.concatenate(tiles, axis=1))
                x = _qwen35_mlp_residual(
                    x_after_attn, w, f"model.layers.{i}", self.cfg, i,
                    self._get_experts,
                    iter_expert_batches=self._iter_expert_batches,
                    profile=profiler)
            mx.eval(x)
            if (tap_collector is not None
                    and i in tap_collector.tap_layers):
                tap_collector.observe(
                    i,
                    x,
                    position_start=(offset if positions3 is None else None),
                    positions=(
                        None if positions3 is None else positions3[0]),
                )
            # ``mx.concatenate`` materializes a new hidden-state array.  The
            # per-tile attention/MLP arrays and the final views into the old
            # hidden state are no longer needed after that evaluation.  Python
            # loop locals otherwise retain one complete prompt-sized layer
            # until the following iteration, needlessly adding hundreds of MB
            # to long-context admission and peak Metal usage.
            tiles.clear()
            xt = None
            yt = None
            mlp_tile = None
            x_after_attn = None
            if profiler is not None and profiler.sync_substeps:
                profiler.record_substep(
                    "mlp", i,
                    dense_mlp_s + time.perf_counter() - mlp_t0,
                    positions=total)
            compute_s = time.perf_counter() - t0
            self.timer.add("layer_compute", compute_s)
            if profiler is not None:
                profiler.record_layer(
                    i, positions=total, weight_wait_s=weight_wait_s,
                    compute_s=compute_s, cache_before=cache_before,
                    cache_after=profiler.cache_snapshot(self.cache),
                    layer_type=self._profile_layer_type(i),
                )
            if on_progress is not None:
                on_progress({
                    "phase": "prefill_layer",
                    "completed_layers": i + 1,
                    "total_layers": n,
                    "total_tokens": total,
                    "cache_source": "cold",
                })
            self._record_layer_transient(
                transient_shape_positions, i,
                _resident_adjusted_transient(
                    active_before, mx.get_active_memory(),
                    mx.get_peak_memory()))
            self._note_true_peak()
            del w
        self._restore_aggregate_layer_transient(transient_shape_positions)
        return x

    def _layer_stationary_qwen4_sweep(
            self, x: mx.array, kv: KVCache, offset: int,
            tile_width: int, on_progress=None) -> mx.array:
        """Bounded-position, layer-major prefill for Qwen4-Exp.

        The 49K released-schema capture makes one four-stream hidden state
        about 1 GiB. Running a whole Qwen4 block on it also materializes PLE,
        hyper-mixer and attention intermediates; the first real request rose
        above 12.8 GiB Metal and was stopped. PLE, QSA, DeltaNet and KV update
        here in causal tile order while each layer trunk remains loaded once.
        Row-local routed MoE arithmetic retains the original tile shapes, but
        its union of physical expert pages is fetched once per complete layer,
        so the 241.6 GB released expert body is not reread per tile.

        The tile arithmetic and complete recurrent endpoint are covered by the
        real greedy/state oracle. The host-spooled 49K memory shape remains an
        explicit candidate until its peak/swap and full-response gates clear.
        """
        from .qwen4_exp import (
            apply_ple,
            hyper_connection_inject,
            hyper_connection_mix,
            qwen4_attention_branch,
        )
        from .expert_batching import consume_expert_batches
        from .glm import _group_routes
        from .layer_runner import _linear, _swiglu
        from .qwen35 import _route_experts
        import numpy as np

        if tile_width <= 0:
            raise ValueError("Qwen4 layer-stationary tile width must be positive")
        total = int(x.shape[1])
        input_ids = tuple(getattr(self, "_qwen4_input_ids", ()))
        if len(input_ids) != total:
            raise ValueError(
                "Qwen4 layer-stationary sweep lost position-aligned input ids")
        profiler = self._request_profiler
        if profiler is not None:
            profiler.begin_sweep(total, path="layer_stationary_qwen4")

        spool_h2d_bytes = 0
        spool_d2h_bytes = 0
        spool_copy_s = 0.0
        spool_peak_host_bytes = 0
        spool_samples = 0
        spool_expert_row_gathers = 0
        spool_expert_rows_uploaded = 0
        spool_expert_batch_tile_gathers = 0
        spool_expert_batch_tile_rows_uploaded = 0
        spool_expert_tile_eval_syncs = 0
        spool_expert_tile_eval_groups = 0
        spool_phase_seconds = {
            "ple": 0.0,
            "attention": 0.0,
            "route_and_spool": 0.0,
            "experts": 0.0,
            "output": 0.0,
        }
        metal_limit_bytes = max(0, int(self.rc.metal_limit_mb)) * 1_000_000
        activation_dtype = x.dtype
        if activation_dtype not in (mx.float16, mx.bfloat16):
            raise TypeError(
                "Qwen4 host spool requires released FP16/BF16 activations, "
                f"got {activation_dtype}")

        def note_spool(
            phase: str, *, layer: int, completed_tokens: int = 0,
            host_bytes: int = 0, publish: bool = True,
        ) -> None:
            """Publish content-blind memory progress and enforce the cap."""
            nonlocal spool_samples
            active = int(mx.get_active_memory())
            peak = int(mx.get_peak_memory())
            observed = max(active, peak)
            spool_samples += 1
            self._note_true_peak()
            progress = {
                "phase": "prefill_layer",
                "diagnostic": "qwen4_host_spool",
                "subphase": phase,
                "layer": int(layer),
                "completed_layers": max(0, int(layer)),
                "total_layers": self.cfg.num_hidden_layers,
                "completed_tokens": int(completed_tokens),
                "total_tokens": total,
                "active_metal_bytes": active,
                "peak_metal_bytes": peak,
                "host_spool_bytes": int(host_bytes),
                "metal_limit_bytes": metal_limit_bytes,
                "cache_source": "cold",
            }
            if publish and on_progress is not None:
                on_progress(progress)
            if metal_limit_bytes and observed > metal_limit_bytes:
                raise MemoryError(
                    "Qwen4 host-spooled prefill crossed its hard Metal cap: "
                    f"phase={phase} layer={layer} tokens={completed_tokens} "
                    f"observed={observed} limit={metal_limit_bytes}")

        def host_bits(value: mx.array) -> np.ndarray:
            """Copy the exact released 16-bit payload without conversion."""
            nonlocal spool_d2h_bytes, spool_copy_s
            started = time.perf_counter()
            result = _lossless_16bit_host_spool(
                value, expected_dtype=activation_dtype)
            spool_d2h_bytes += int(result.nbytes)
            spool_copy_s += time.perf_counter() - started
            return result

        def metal_bits(value: np.ndarray) -> mx.array:
            nonlocal spool_h2d_bytes, spool_copy_s
            started = time.perf_counter()
            result = _restore_lossless_16bit_host_spool(
                value, dtype=activation_dtype)
            spool_h2d_bytes += int(value.nbytes)
            spool_copy_s += time.perf_counter() - started
            return result

        # One prompt-sized host copy is intentionally retained while Metal
        # owns only bounded tiles. On this model the four-stream 49K hidden is
        # ~1.01GB; retaining all seven Metal tiles and their lazy parents was
        # measured at 14.4GB. Host uint16 preserves every released 16-bit bit.
        # Each causal tile's input is dead after its attention/recurrent state
        # update, so the same host allocation is overwritten first with the
        # post-attention bits and later with the final layer-output bits. This
        # avoids a second prompt-sized host array without changing arithmetic,
        # tile boundaries, or any activation payload.
        hidden_host = host_bits(x)
        spool_peak_host_bytes = int(hidden_host.nbytes)
        note_spool(
            "initial_hidden", layer=0, completed_tokens=total,
            host_bytes=spool_peak_host_bytes)

        for i in range(self.cfg.num_hidden_layers):
            self._select_layer_transient(total, i)
            cache_before = (
                profiler.cache_snapshot(self.cache)
                if profiler is not None else None)
            layer_key = self._layer_key(i)
            layer_names = self._layer_names(i)
            wait_t0 = time.perf_counter()
            if not self.cache.contains(layer_key):
                incoming_page = self._layer_fetch_bytes_estimate(i)
                if incoming_page:
                    self.cache.prepare_for(incoming_page)
                    if self.governor is not None:
                        self.governor.reserve(
                            incoming_page, reason="qwen4-prefill-layer-page")
            w = self.cache.get(layer_key, layer_names)
            weight_wait_s = time.perf_counter() - wait_t0
            self.timer.add("weights_wait", weight_wait_s)

            active_before = mx.get_active_memory()
            mx.reset_peak_memory()
            compute_t0 = time.perf_counter()
            prefix = f"model.layers.{i}"

            if (i in self.cfg.qwen4_ple_layers
                    and self._qwen4_ple_rows is None):
                raise ValueError(
                    "Qwen4 layer-stationary PLE is missing its row store")

            # Preserve every row-local operator's original tile shape. Spill
            # evaluated intermediates to host BF16 bits between phases so no
            # prompt-sized MLX graph/list survives across tiles.
            mixed_host = np.empty(
                (1, total, self.cfg.hidden_size), dtype=np.uint16)
            injection_host = np.empty(
                (1, total, self.cfg.qwen4_hc_count), dtype=np.uint16)
            spool_peak_host_bytes = max(
                spool_peak_host_bytes,
                int(hidden_host.nbytes
                    + mixed_host.nbytes + injection_host.nbytes))
            records = []
            global_positions: dict[int, list[int]] = {}
            global_routes: dict[int, list[tuple[int, float]]] = {}
            for pos in range(0, total, tile_width):
                end = min(pos + tile_width, total)
                source = metal_bits(hidden_host[:, pos:end])
                phase_started = time.perf_counter()
                if i in self.cfg.qwen4_ple_layers:
                    ple = apply_ple(
                        source, input_ids[pos:end], w, f"{prefix}.ple",
                        self.cfg, i, self._qwen4_ple_rows,
                        getattr(kv, "qwen4_cache", None))
                    source = source + ple
                    mx.eval(source)
                    spool_phase_seconds["ple"] += (
                        time.perf_counter() - phase_started)
                phase_started = time.perf_counter()
                mixed, hyper_input, injection = hyper_connection_mix(
                    source, w, f"{prefix}.attn_hyper_connection", self.cfg)
                mx.eval(mixed, injection)
                branch = qwen4_attention_branch(
                    mixed, w, prefix, self.cfg, kv, i, offset + pos,
                    compiled_delta_prefill=(
                        self.rc.qwen_compiled_delta_prefill),
                    native_fused_delta_prefill=(
                        self.rc.qwen_native_fused_delta_prefill))
                post_attention = hyper_connection_inject(
                    branch, hyper_input, injection)
                mx.eval(post_attention)
                spool_phase_seconds["attention"] += (
                    time.perf_counter() - phase_started)
                phase_started = time.perf_counter()
                mixed, hyper_input, injection = hyper_connection_mix(
                    post_attention, w,
                    f"{prefix}.mlp_hyper_connection", self.cfg)
                indices, scores = _route_experts(
                    mixed, w, prefix, self.cfg, i)
                mx.eval(mixed, injection, indices, scores)
                groups = _group_routes(indices, scores)
                # ``source`` has been evaluated and the causal state for this
                # tile has already been committed. No later tile reads the old
                # host slice, so replace it with the exact post-attention bits.
                hidden_host[:, pos:end] = host_bits(post_attention)
                mixed_host[:, pos:end] = host_bits(mixed)
                injection_host[:, pos:end] = host_bits(injection)
                for expert, rows in groups.items():
                    global_positions.setdefault(int(expert), []).extend(
                        pos + int(row) for row, _weight in rows)
                    global_routes.setdefault(int(expert), []).extend(
                        (pos + int(row), float(weight))
                        for row, weight in rows)
                records.append({
                    "start": pos,
                    "end": end,
                    "groups": groups,
                    "routed": mx.zeros(
                        (1, end - pos, self.cfg.hidden_size),
                        dtype=activation_dtype),
                })
                source = ple = mixed = hyper_input = injection = None
                branch = post_attention = indices = scores = None
                mx.clear_cache()
                spool_phase_seconds["route_and_spool"] += (
                    time.perf_counter() - phase_started)
                note_spool(
                    "attention_tile", layer=i, completed_tokens=end,
                    host_bytes=spool_peak_host_bytes,
                    publish=(pos == 0 or end == total
                             or end % (tile_width * 8) == 0))

            expert_ids = sorted(global_positions)

            def archive_only_prefill_batches():
                prior = self.store.raw_fast_tier_enabled
                self.store.raw_fast_tier_enabled = False
                try:
                    yield from self._iter_expert_batches(
                        i, expert_ids, positions=global_positions)
                finally:
                    self.store.raw_fast_tier_enabled = prior

            batches = (
                archive_only_prefill_batches()
                if self.rc.qwen4_fast_tier_decode_only
                else self._iter_expert_batches(
                    i, expert_ids, positions=global_positions)
            )
            expert_batches_done = 0

            def consume_batch(batch_ids, experts):
                nonlocal expert_batches_done
                nonlocal spool_expert_row_gathers
                nonlocal spool_expert_rows_uploaded
                nonlocal spool_expert_batch_tile_gathers
                nonlocal spool_expert_batch_tile_rows_uploaded
                nonlocal spool_expert_tile_eval_syncs
                nonlocal spool_expert_tile_eval_groups
                if self.rc.qwen4_global_expert_rows:
                    # For a long prompt, virtually every 16-expert page batch
                    # touches every position at least once. The baseline thus
                    # uploads the complete mixed tile for all 32 batches. This
                    # schedule gathers only the rows routed to one expert and
                    # evaluates that expert once over the complete prompt.
                    # Routes retain tile order, and each per-tile accumulator
                    # is updated in the same ascending expert-id order.
                    for expert in batch_ids:
                        routes = global_routes.get(int(expert), ())
                        if not routes:
                            continue
                        positions = [position for position, _ in routes]
                        mixed_rows = metal_bits(mixed_host[:, positions])
                        route_weights = mx.array(
                            [weight for _, weight in routes],
                            dtype=mixed_rows.dtype)
                        expert_prefix = f"{prefix}.mlp.experts.{expert}"
                        contribution = _swiglu(
                            mixed_rows, experts[expert], expert_prefix)
                        weighted = (
                            contribution * route_weights[None, :, None])
                        mx.eval(weighted)
                        spool_expert_row_gathers += 1
                        spool_expert_rows_uploaded += len(positions)

                        cursor = 0
                        touched = []
                        for record in records:
                            rows = record["groups"].get(expert)
                            if not rows:
                                continue
                            count = len(rows)
                            local_positions = [
                                int(position) for position, _ in rows]
                            record["routed"] = record["routed"].at[
                                :, local_positions, :].add(
                                    weighted[:, cursor:cursor + count, :])
                            touched.append(record["routed"])
                            cursor += count
                        if cursor != len(routes):
                            raise RuntimeError(
                                "Qwen4 global expert route gather lost rows")
                        if touched:
                            mx.eval(*touched)
                        mixed_rows = route_weights = None
                        contribution = weighted = touched = None
                        mx.clear_cache()
                    expert_batches_done += 1
                    if (expert_batches_done == 1
                            or expert_batches_done % 8 == 0):
                        note_spool(
                            "expert_batch", layer=i, completed_tokens=total,
                            host_bytes=spool_peak_host_bytes)
                    return
                # Evaluate one tile before loading the next. A union batch's
                # released pages remain shared, while routed accumulation has
                # the same expert-id order and tile shapes as chunk-major.
                pending_routed = []
                eval_batch = max(
                    1, int(self.rc.qwen4_expert_tile_eval_batch))
                for record in records:
                    start, end = record["start"], record["end"]
                    if self.rc.qwen4_sparse_expert_batch_rows:
                        batch_positions = sorted({
                            int(position)
                            for expert in batch_ids
                            for position, _weight in (
                                record["groups"].get(expert) or ())
                        })
                        if not batch_positions:
                            continue
                        compact_position = {
                            position: compact
                            for compact, position in enumerate(batch_positions)
                        }
                        mixed_tile = metal_bits(
                            mixed_host[:, start:end][:, batch_positions])
                        spool_expert_batch_tile_gathers += 1
                        spool_expert_batch_tile_rows_uploaded += len(
                            batch_positions)
                    else:
                        compact_position = None
                        mixed_tile = metal_bits(mixed_host[:, start:end])
                    for expert in batch_ids:
                        rows = record["groups"].get(expert)
                        if not rows:
                            continue
                        expert_prefix = f"{prefix}.mlp.experts.{expert}"
                        positions = [int(position) for position, _ in rows]
                        source_positions = (
                            positions if compact_position is None else
                            [compact_position[position]
                             for position in positions])
                        route_weights = mx.array(
                            [weight for _, weight in rows],
                            dtype=mixed_tile.dtype)
                        contribution = _swiglu(
                            mixed_tile[:, source_positions],
                            experts[expert], expert_prefix)
                        record["routed"] = record["routed"].at[
                            :, positions, :].add(
                                contribution * route_weights[None, :, None])
                    pending_routed.append(record["routed"])
                    if len(pending_routed) >= eval_batch:
                        mx.eval(*pending_routed)
                        spool_expert_tile_eval_syncs += 1
                        spool_expert_tile_eval_groups += len(pending_routed)
                        pending_routed.clear()
                        mx.clear_cache()
                    mixed_tile = contribution = route_weights = None
                if pending_routed:
                    mx.eval(*pending_routed)
                    spool_expert_tile_eval_syncs += 1
                    spool_expert_tile_eval_groups += len(pending_routed)
                    pending_routed.clear()
                    mx.clear_cache()
                expert_batches_done += 1
                if expert_batches_done == 1 or expert_batches_done % 8 == 0:
                    note_spool(
                        "expert_batch", layer=i, completed_tokens=total,
                        host_bytes=spool_peak_host_bytes)

            phase_started = time.perf_counter()
            # The layer transient learned above is the attention/GDN tile
            # peak (4.9GB on the real 49K capture). Those graphs have already
            # been evaluated, copied to host, and cleared before routed expert
            # pages are fetched; charging that dead phase to every expert
            # batch produced a false 3.56GB reservation and aborted at layer
            # four while live Metal was only 1.62GB. Bound only the incremental
            # expert-phase scratch here. The 640MB allowance exceeds the
            # measured active rise (~411MB) and the released 16-expert BF16
            # page payload (~157MB); the hard 8.5GB sampler remains in force.
            attention_transient = self._layer_transient
            self._layer_transient = 640_000_000
            try:
                consume_expert_batches(batches, consume_batch)
            finally:
                self._layer_transient = attention_transient
            spool_phase_seconds["experts"] += (
                time.perf_counter() - phase_started)

            phase_started = time.perf_counter()
            for record in records:
                start, end = record["start"], record["end"]
                mixed = metal_bits(mixed_host[:, start:end])
                hyper_input = metal_bits(hidden_host[:, start:end])
                injection = metal_bits(injection_host[:, start:end])
                shared = _swiglu(
                    mixed, w, f"{prefix}.mlp.shared_expert")
                shared_gate = mx.sigmoid(_linear(
                    mixed, w, f"{prefix}.mlp.shared_expert_gate"))
                branch = (
                    record["routed"] + shared_gate * shared
                ).astype(activation_dtype)
                output = hyper_connection_inject(
                    branch, hyper_input, injection)
                mx.eval(output)
                # The post-attention slice has been copied into Metal and is
                # no longer needed after injection; replace it in place with
                # the exact final layer-output bits.
                hidden_host[:, start:end] = host_bits(output)
                mixed = hyper_input = injection = None
                shared = shared_gate = branch = output = None
                record["routed"] = None
                mx.clear_cache()
                note_spool(
                    "output_tile", layer=i, completed_tokens=end,
                    host_bytes=spool_peak_host_bytes,
                    publish=(start == 0 or end == total
                             or end % (tile_width * 8) == 0))
            spool_phase_seconds["output"] += (
                time.perf_counter() - phase_started)
            mixed_host = injection_host = None
            records.clear()

            compute_s = time.perf_counter() - compute_t0
            self.timer.add("layer_compute", compute_s)
            if profiler is not None:
                profiler.record_layer(
                    i, positions=total, weight_wait_s=weight_wait_s,
                    compute_s=compute_s, cache_before=cache_before,
                    cache_after=profiler.cache_snapshot(self.cache),
                    layer_type=self._profile_layer_type(i))
            if on_progress is not None:
                on_progress({
                    "phase": "prefill_layer",
                    "completed_layers": i + 1,
                    "total_layers": self.cfg.num_hidden_layers,
                    "total_tokens": total,
                    "cache_source": "cold",
                })
            note_spool(
                "layer_complete", layer=i + 1, completed_tokens=total,
                host_bytes=spool_peak_host_bytes)
            self._record_layer_transient(
                total, i,
                _resident_adjusted_transient(
                    active_before, mx.get_active_memory(),
                    mx.get_peak_memory()))
            self._note_true_peak()
            del w
            mx.clear_cache()
        self._restore_aggregate_layer_transient(total)
        self._qwen4_host_spool_stats = {
            "h2d_bytes": spool_h2d_bytes,
            "d2h_bytes": spool_d2h_bytes,
            "copy_seconds": spool_copy_s,
            "peak_host_bytes": spool_peak_host_bytes,
            "memory_samples": spool_samples,
            "activation_dtype": str(activation_dtype),
            "global_expert_rows": int(self.rc.qwen4_global_expert_rows),
            "expert_row_gathers": spool_expert_row_gathers,
            "expert_rows_uploaded": spool_expert_rows_uploaded,
            "sparse_expert_batch_rows": int(
                self.rc.qwen4_sparse_expert_batch_rows),
            "expert_batch_tile_gathers": spool_expert_batch_tile_gathers,
            "expert_batch_tile_rows_uploaded": (
                spool_expert_batch_tile_rows_uploaded),
            "expert_tile_eval_batch": int(
                self.rc.qwen4_expert_tile_eval_batch),
            "expert_tile_eval_syncs": spool_expert_tile_eval_syncs,
            "expert_tile_eval_groups": spool_expert_tile_eval_groups,
            "fast_tier_decode_only": int(
                self.rc.qwen4_fast_tier_decode_only),
            **{
                f"{name}_seconds": round(seconds, 6)
                for name, seconds in spool_phase_seconds.items()
            },
        }
        # All causal state covers the complete prompt; only the final hidden
        # row is consumed by the output mixer/head at this endpoint.
        self._qwen4_input_ids = (input_ids[-1],)
        return metal_bits(hidden_host[:, -1:])

    def _layer_stationary_gptoss_sweep(
            self, x: mx.array, kv: KVCache, offset: int,
            tile_width: int, on_progress=None) -> mx.array:
        """Layer-major GPT-OSS prefill with bounded causal tiles and experts.

        GPT-OSS's ordinary safe prefill is chunk-major: every 64--512 position
        chunk streams all 36 out-of-core layer trunks again.  This inverse loop
        fetches one layer, advances only that layer's released attention/KV in
        the same causal tile order, concatenates the post-attention rows, and
        evaluates the stateless router/MoE once for the complete range.  Each
        layer page is therefore fetched once rather than once per prompt chunk.

        The attention and MLP halves are the exact helpers used by
        ``run_gptoss_block``. Expert contributions retain their historical
        insertion order, while ``_iter_expert_batches`` materializes each
        bounded page group before the next is fetched.  The path remains
        explicit because full-model greedy equivalence and broad real-request
        evidence are required before changing GPT-OSS's default schedule.
        """
        from .gptoss import (
            _gptoss_attention_residual, _gptoss_tiled_mlp_residual)

        if tile_width <= 0:
            raise ValueError("tile_width must be positive")
        n = self.cfg.num_hidden_layers
        total = int(x.shape[1])
        (self._layer_transient,
         self._layer_transient_margin) = _layer_transient_for_positions(
             total,
             getattr(
                 self, "_prefill_layer_transient_by_positions", {}
             ).get(total, 0),
             getattr(self, "_decode_layer_transient", 0))
        profiler = self._request_profiler
        if profiler is not None:
            profiler.begin_sweep(total, path="layer_stationary_gptoss")
        for i in range(n):
            self._select_layer_transient(total, i)
            if self.prefetcher:
                for j in range(
                        i + 1, min(i + 1 + self.rc.prefetch_depth, n)):
                    self.prefetcher.schedule(
                        self._layer_key(j), self._layer_names(j))

            cache_before = (
                profiler.cache_snapshot(self.cache)
                if profiler is not None else None)
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
            if self._dsv4_packed_trunk:
                w = self._materialize_packed_trunk(w)
            weight_wait_s = time.perf_counter() - t0
            self.timer.add("weights_wait", weight_wait_s)

            active_before = mx.get_active_memory()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            tiles = []
            pos = 0
            while pos < total:
                end = min(pos + tile_width, total)
                if self.governor is not None and self._layer_transient:
                    self.governor.reserve(
                        self._layer_transient,
                        margin=self._layer_transient_margin)
                attention_t0 = time.perf_counter()
                yt = _gptoss_attention_residual(
                    x[:, pos:end, :], w, f"model.layers.{i}", self.cfg,
                    kv, i, offset + pos, self._rope_freqs, self._mscale)
                mx.eval(yt)
                if profiler is not None and profiler.sync_substeps:
                    profiler.record_substep(
                        "attention", i,
                        time.perf_counter() - attention_t0,
                        positions=end - pos)
                tiles.append(yt)
                pos = end
            mlp_t0 = time.perf_counter()
            x = _gptoss_tiled_mlp_residual(
                tiles, w, f"model.layers.{i}", self.cfg, i,
                self._get_experts,
                iter_expert_batches=self._iter_expert_batches,
                profile=profiler)
            mx.eval(x)
            if profiler is not None and profiler.sync_substeps:
                profiler.record_substep(
                    "mlp", i, time.perf_counter() - mlp_t0,
                    positions=total)
            compute_s = time.perf_counter() - t0
            self.timer.add("layer_compute", compute_s)
            if profiler is not None:
                profiler.record_layer(
                    i, positions=total, weight_wait_s=weight_wait_s,
                    compute_s=compute_s, cache_before=cache_before,
                    cache_after=profiler.cache_snapshot(self.cache),
                    layer_type=self._profile_layer_type(i))
            if on_progress is not None:
                on_progress({
                    "phase": "prefill_layer",
                    "completed_layers": i + 1,
                    "total_layers": n,
                    "total_tokens": total,
                    "cache_source": "cold",
                })
            self._record_layer_transient(
                total, i,
                _resident_adjusted_transient(
                    active_before, mx.get_active_memory(),
                    mx.get_peak_memory()))
            self._note_true_peak()
            del w
        self._restore_aggregate_layer_transient(total)
        return x

    def _qwen35_lossy_suffix_prefill_sweep(
            self, x: mx.array, kv: KVCache, offset: int,
            tile_width: int, on_progress=None, *,
            stable_boundary_tokens: int | None = None,
            boundary_fork_kv: KVCache | None = None) -> mx.array:
        """Run the fixed mixed-depth Qwen endpoint-packed prefill schedule.

        Every prompt position crosses the first ``early_layers`` blocks, so
        lower recurrent/full-attention state and retained hidden inputs are
        derived from the complete prompt. A fixed leading ``prefix_tokens``
        anchor and final ``suffix_tokens`` continue through the upper blocks.
        This gives upper layers both early instruction/schema semantics and
        recent conversational intent without classifying prompt contents.
        Upper recurrence skips the unretained middle and upper full-attention
        KV contains only the packed endpoints. RoPE retains every endpoint's
        original global position.

        This is deliberately lossy, fixed by configuration rather than request
        contents, and used only by server.py's explicit fast-profile opt-in.
        """
        early_layers = int(
            self.rc.qwen_lossy_suffix_prefill_early_layers)
        prefix_tokens = int(
            self.rc.qwen_lossy_suffix_prefill_prefix_tokens)
        suffix_tokens = int(self.rc.qwen_lossy_suffix_prefill_tokens)
        total_layers = int(self.cfg.num_hidden_layers)
        total_tokens = int(x.shape[1])
        if not 0 < early_layers < total_layers:
            raise ValueError("invalid Qwen lossy suffix-prefill layer depth")
        if prefix_tokens < 0 or suffix_tokens <= 0:
            raise ValueError("invalid Qwen lossy suffix-prefill token count")
        retained_tokens = prefix_tokens + suffix_tokens
        stable_tokens = (
            total_tokens
            if stable_boundary_tokens is None
            else int(stable_boundary_tokens))
        if not 0 < stable_tokens <= total_tokens:
            raise ValueError("invalid Qwen lossy stable-boundary token count")
        if boundary_fork_kv is not None and stable_tokens >= total_tokens:
            raise ValueError(
                "a Qwen boundary fork requires a following scaffold")
        boundary_fork_at = (
            stable_tokens if boundary_fork_kv is not None else None)
        if stable_tokens <= retained_tokens:
            return self._layer_stationary_qwen35_sweep(
                x, kv, offset=offset, tile_width=tile_width,
                on_progress=on_progress,
                boundary_fork_at=boundary_fork_at,
                boundary_fork_kv=boundary_fork_kv)

        x = self._layer_stationary_qwen35_sweep(
            x, kv, offset=offset, tile_width=tile_width,
            on_progress=on_progress, layer_end=early_layers,
            profile_path="qwen35_lossy_suffix_shallow",
            boundary_fork_at=boundary_fork_at,
            boundary_fork_kv=boundary_fork_kv)
        # When a generation scaffold follows the stable chat boundary, retain
        # the configured anchor/suffix from the stable portion and append the
        # scaffold. Selecting the suffix from ``total_tokens`` instead would
        # silently replace recent user/tool evidence with scaffold tokens.
        suffix_start = stable_tokens - suffix_tokens
        scaffold = x[:, stable_tokens:, :]
        if prefix_tokens:
            x = mx.concatenate(
                (x[:, :prefix_tokens, :],
                 x[:, suffix_start:stable_tokens, :], scaffold), axis=1)
            positions = mx.concatenate((
                mx.arange(offset, offset + prefix_tokens, dtype=mx.float32),
                mx.arange(
                    offset + suffix_start, offset + stable_tokens,
                    dtype=mx.float32),
                mx.arange(
                    offset + stable_tokens, offset + total_tokens,
                    dtype=mx.float32),
            ))
            positions3 = mx.stack((positions, positions, positions), axis=0)
            deep_offset = offset
            profile_path = "qwen35_lossy_endpoint_packed_deep"
        else:
            x = x[:, suffix_start:, :]
            positions3 = None
            deep_offset = offset + suffix_start
            profile_path = "qwen35_lossy_suffix_deep"
        return self._layer_stationary_qwen35_sweep(
            x, kv, offset=deep_offset, tile_width=tile_width,
            on_progress=on_progress, layer_start=early_layers,
            profile_path=profile_path, positions3=positions3,
            boundary_fork_at=(
                retained_tokens if boundary_fork_kv is not None else None),
            boundary_fork_kv=boundary_fork_kv)

    def _layer_stationary_glm_sweep(
            self, x: mx.array, kv, offset: int, tile_width: int,
            on_progress=None) -> mx.array:
        """F35 extension (2026-07-25): layer-major prefill for GLM-5.2/K2.5/
        glm4_moe_lite (real q_lora MLA + noaux_tc MoE, run_glm_block's
        shape) -- the same technique `_layer_stationary_kimi_linear_sweep`
        applies to Kimi Linear's MLA layers, reusing the identical
        attention/MLP split pattern (`_glm_attention_residual` /
        `_glm_mlp_residual` in runtime/glm.py, split out of run_glm_block the
        same way `_kimi_linear_attention_residual`/`_kimi_linear_mlp_residual`
        were split out of run_kimi_linear_block).

        MLA attention has no recurrent state analogous to KDA, but the same
        correctness argument still applies: each layer's attention depends
        only on that layer's own KV history and current tile, never on
        another layer's loop position, and MoE routing is a stateless
        function of `h` -- computing it once per layer over the whole tiled
        range is the same function evaluated on the union of what chunk-major
        would have called it on per-chunk, not a different function. Proven
        directly in tests/test_f35_glm_layer_stationary_oracle.py against the
        real Kimi-K2.5 checkpoint (kimi_k25 shares this exact block shape).
        """
        from .glm import _glm_attention_residual, _glm_mlp_residual
        import numpy as np

        if tile_width <= 0:
            raise ValueError("tile_width must be positive")
        n = self.cfg.num_hidden_layers
        total = int(x.shape[1])
        host_spool = bool(getattr(
            self.rc, "glm53_layer_stationary_host_spool", False))
        spool_h2d_bytes = 0
        spool_d2h_bytes = 0
        spool_copy_s = 0.0
        spool_peak_host_bytes = 0
        memory_samples = 0
        memory_peak = 0
        transient_reservation_calls = 0
        transient_reservation_bytes = 0
        transient_reservation_margin_bytes = 0
        transient_reservation_s = 0.0
        transient_reservation_first_margin_calls = 0
        transient_reservation_recurring_calls = 0
        memory_phase_active = {
            "initial_carrier": 0,
            "attention": 0,
            "ffn_hc_pre": 0,
            "mlp": 0,
            "ffn_hc_post": 0,
        }
        metal_limit_bytes = max(
            0, int(getattr(self.rc, "metal_limit_mb", 0))) * 1_000_000

        def host_bits(value: mx.array, *, retained_host_bytes: int = 0):
            """Copy released BF16 bits to CPU-owned memory without casting."""
            nonlocal spool_d2h_bytes, spool_copy_s, spool_peak_host_bytes
            if value.dtype != mx.bfloat16:
                raise TypeError(
                    "full GLM host spool requires BF16 activations, "
                    f"got {value.dtype}")
            started = time.perf_counter()
            mx.eval(value)
            result = np.array(
                np.asarray(value.view(mx.uint16)), dtype=np.uint16, copy=True)
            spool_d2h_bytes += int(result.nbytes)
            spool_copy_s += time.perf_counter() - started
            spool_peak_host_bytes = max(
                spool_peak_host_bytes,
                int(retained_host_bytes) + int(result.nbytes))
            return result

        def metal_bits(value: np.ndarray) -> mx.array:
            """Restore one raw BF16 CPU payload at its original shape."""
            nonlocal spool_h2d_bytes, spool_copy_s
            started = time.perf_counter()
            result = mx.array(value, dtype=mx.uint16).view(mx.bfloat16)
            mx.eval(result)
            spool_h2d_bytes += int(value.nbytes)
            spool_copy_s += time.perf_counter() - started
            return result

        def note_memory(
                phase: str, layer: int, completed_tokens: int = 0,
                *, publish: bool = False) -> None:
            """Attribute full-GLM phases and enforce the configured Metal cap."""
            nonlocal memory_samples, memory_peak
            active = int(mx.get_active_memory())
            peak = int(mx.get_peak_memory())
            observed = max(active, peak)
            if phase.startswith("attention") or phase == "preselect":
                group = "attention"
            elif phase.startswith("dense_mlp") or phase == "routed_mlp":
                group = "mlp"
            else:
                group = "initial_carrier"
            memory_phase_active[group] = max(
                memory_phase_active[group], active)
            memory_samples += 1
            memory_peak = max(memory_peak, observed)
            self._note_true_peak()
            if publish and on_progress is not None:
                on_progress({
                    "phase": "prefill_layer",
                    "diagnostic": "glm53_layer_stationary",
                    "subphase": phase,
                    "layer": int(layer),
                    "completed_layers": max(0, int(layer)),
                    "total_layers": n,
                    "completed_tokens": int(completed_tokens),
                    "total_tokens": total,
                    "active_metal_bytes": active,
                    "peak_metal_bytes": peak,
                    "host_spool_bytes": int(spool_peak_host_bytes),
                    "metal_limit_bytes": metal_limit_bytes,
                    "cache_source": "cold",
                })
            if metal_limit_bytes and observed > metal_limit_bytes:
                raise MemoryError(
                    "full GLM layer-stationary prefill crossed its hard Metal "
                    f"cap: phase={phase} layer={layer} "
                    f"tokens={completed_tokens} observed={observed} "
                    f"limit={metal_limit_bytes}")

        hidden_host = None
        if host_spool:
            hidden_host = host_bits(x)
            del x
            mx.clear_cache()
            note_memory("initial_carrier", 0, total, publish=True)
        else:
            note_memory("initial_carrier", 0, total)
        (self._layer_transient,
         self._layer_transient_margin) = _layer_transient_for_positions(
             total,
             getattr(
                 self, "_prefill_layer_transient_by_positions", {}
             ).get(total, 0),
             getattr(self, "_decode_layer_transient", 0))
        profiler = self._request_profiler
        if profiler is not None:
            profiler.begin_sweep(total, path="layer_stationary_glm")
        dsa = getattr(kv, "dsa", None)
        if (
            self.cfg.model_type == "glm_moe_dsa"
            and dsa is not None
            and self.rc.glm_dsa_index_preallocate
        ):
            dsa.set_index_capacity_hint(offset + total)
        for i in range(n):
            self._select_layer_transient(total, i)
            if self.prefetcher:
                for j in range(i + 1, min(i + 1 + self.rc.prefetch_depth, n)):
                    self.prefetcher.schedule(self._layer_key(j), self._layer_names(j))

            cache_before = (
                profiler.cache_snapshot(self.cache)
                if profiler is not None else None)
            t0 = time.perf_counter()
            layer_key = self._layer_key(i)
            layer_names = self._layer_names(i)
            if not self.cache.contains(layer_key):
                incoming_page = self._layer_fetch_bytes_estimate(i)
                if incoming_page:
                    self.cache.prepare_for(incoming_page)
                    if self.governor is not None:
                        self.governor.reserve(
                            incoming_page,
                            reason="glm53-full-prefill-layer-page")
            w = self.cache.get(layer_key, layer_names)
            if self._dsv4_packed_trunk:
                w = self._materialize_packed_trunk(w)
            weight_wait_s = time.perf_counter() - t0
            self.timer.add("weights_wait", weight_wait_s)

            if host_spool:
                # Keep the prompt-wide hidden state off Metal while the next
                # released weight page is admitted and fetched. Restore it
                # ONCE for the layer so all original tile views, concatenates,
                # and GEMM layouts remain unchanged. Restoring every tile as a
                # fresh contiguous allocation failed the real tile-32 identity
                # gate even though its BF16 payloads were bit-exact.
                x = metal_bits(hidden_host)
                hidden_host = None

            active_before = mx.get_active_memory()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            dsa = getattr(kv, "dsa", None)
            indexer_type = (
                self.cfg.indexer_types[i]
                if self.cfg.indexer_types
                and i < len(self.cfg.indexer_types)
                else "shared"
            )
            selection_width = int(
                getattr(dsa, "selection_query_tile_size", 0) or 0)
            if (
                self.cfg.model_type == "glm_moe_dsa"
                and dsa is not None
                and indexer_type == "full"
                and selection_width > tile_width
            ):
                dsa.preselect_full_layer(
                    i, x, w, f"model.layers.{i}", offset,
                    attention_tile_width=tile_width)
                # All query-sized selections are now external-spilled; return
                # score graphs/allocator cache before compact MLA expansion.
                mx.clear_cache()
                note_memory("preselect", i, total, publish=host_spool)
            is_dense_mlp = (
                self.cfg.mlp_layer_types[i] == "dense"
                if i < len(self.cfg.mlp_layer_types)
                else i < self.cfg.first_k_dense_replace
            )
            dense_mlp_tile_size = int(
                self.rc.glm_dsa_dense_mlp_tile_size or 0)
            tile_dense_mlp = bool(
                is_dense_mlp and dense_mlp_tile_size > 0
                and total > dense_mlp_tile_size)
            tiles = []
            dense_attention_tiles = []
            dense_attention_positions = 0
            dense_mlp_s = 0.0
            pos = 0
            while pos < total:
                end = min(pos + tile_width, total)
                if self.governor is not None and self._layer_transient:
                    signature = self._transient_layer_signature(i)
                    observations = int(getattr(
                        self, "_layer_transient_observation_counts", {}
                    ).get((total, signature), 0))
                    reserve_margin = (
                        _recurring_layer_transient_reserve_margin(
                            total, observations))
                    reservation_t0 = time.perf_counter()
                    transient_reservation_calls += 1
                    transient_reservation_bytes += int(self._layer_transient)
                    transient_reservation_margin_bytes += int(reserve_margin)
                    transient_reservation_first_margin_calls += int(
                        reserve_margin > 0)
                    transient_reservation_recurring_calls += int(
                        reserve_margin == 0)
                    try:
                        self.governor.reserve(
                            self._layer_transient,
                            margin=reserve_margin,
                            reason="glm53-full-attention-transient")
                    except MemoryError:
                        print(
                            "[glm53-full-prefill-admission] "
                            f"layer={i} signature={signature} "
                            f"tile={end - pos} position={pos}/{total} "
                            f"active={int(mx.get_active_memory())} "
                            f"scratch={int(self._layer_transient)} "
                            f"observations={observations} "
                            f"margin={int(reserve_margin)}",
                            flush=True,
                        )
                        raise
                    finally:
                        transient_reservation_s += (
                            time.perf_counter() - reservation_t0)
                xt = x[:, pos:end, :]
                attention_t0 = time.perf_counter()
                yt = _glm_attention_residual(
                    xt, w, f"model.layers.{i}", self.cfg, kv, i, offset + pos,
                    mlp_last_only=False)
                mx.eval(yt)
                if profiler is not None and profiler.sync_substeps:
                    profiler.record_substep(
                        "attention", i,
                        time.perf_counter() - attention_t0,
                        positions=end - pos)
                if tile_dense_mlp:
                    dense_attention_tiles.append(yt)
                    dense_attention_positions += end - pos
                    if (dense_attention_positions >= dense_mlp_tile_size
                            or end == total):
                        dense_input = (
                            dense_attention_tiles[0]
                            if len(dense_attention_tiles) == 1
                            else mx.concatenate(
                                dense_attention_tiles, axis=1))
                        dense_mlp_t0 = time.perf_counter()
                        dense_output = _glm_mlp_residual(
                            dense_input, w, f"model.layers.{i}", self.cfg, i,
                            self._get_experts,
                            iter_expert_batches=self._iter_expert_batches,
                            profile=profiler)
                        mx.eval(dense_output)
                        dense_mlp_s += time.perf_counter() - dense_mlp_t0
                        tiles.append(dense_output)
                        dense_attention_tiles = []
                        dense_attention_positions = 0
                else:
                    tiles.append(yt)
                pos = end
            mlp_t0 = time.perf_counter()
            if tile_dense_mlp:
                if dense_attention_tiles or dense_attention_positions:
                    raise AssertionError(
                        "dense GLM MLP tile buffer was not flushed")
                x = (
                    tiles[0]
                    if len(tiles) == 1
                    else mx.concatenate(tiles, axis=1))
            else:
                x_after_attn = (
                    tiles[0]
                    if len(tiles) == 1
                    else mx.concatenate(tiles, axis=1))
                x = _glm_mlp_residual(
                    x_after_attn, w, f"model.layers.{i}", self.cfg, i,
                    self._get_experts,
                    iter_expert_batches=self._iter_expert_batches,
                    profile=profiler)
            mx.eval(x)
            note_memory("routed_mlp", i, total, publish=host_spool)
            if host_spool and i + 1 < n:
                hidden_host = host_bits(x)
                del x
                if not tile_dense_mlp:
                    del x_after_attn
                mx.clear_cache()
            if (
                self.cfg.model_type == "glm_moe_dsa"
                and self.rc.glm_dsa_mla_kv_spill_dir
                and getattr(kv, "latent_spill_enabled", False)
                and kv.spill_latent_layer(i)
            ):
                # This layer's released-dtype latent is not consumed by any
                # later prefill layer. Decode restores exactly one layer at a
                # time; keeping all 78 resident would cost ~4.4 GB at 49K.
                mx.clear_cache()
            if profiler is not None and profiler.sync_substeps:
                profiler.record_substep(
                    "mlp", i,
                    dense_mlp_s + time.perf_counter() - mlp_t0,
                    positions=total)
            compute_s = time.perf_counter() - t0
            self.timer.add("layer_compute", compute_s)
            if profiler is not None:
                profiler.record_layer(
                    i, positions=total, weight_wait_s=weight_wait_s,
                    compute_s=compute_s, cache_before=cache_before,
                    cache_after=profiler.cache_snapshot(self.cache),
                    layer_type=self._profile_layer_type(i),
                )
            if on_progress is not None:
                on_progress({
                    "phase": "prefill_layer",
                    "completed_layers": i + 1,
                    "total_layers": n,
                    "total_tokens": total,
                    "cache_source": "cold",
                })
            self._record_layer_transient(
                total, i,
                _resident_adjusted_transient(
                    active_before, mx.get_active_memory(),
                    mx.get_peak_memory()))
            self._note_true_peak()
            del w
        dsa = getattr(kv, "dsa", None)
        if dsa is not None:
            # The final full indexer layer's per-query IndexShare selections
            # have no remaining prefill consumer. Decode recomputes its own
            # length-one selection and should not retain ~400 MB of tile state.
            clear_selections = getattr(dsa, "clear_selections", None)
            if callable(clear_selections):
                clear_selections()
        self._restore_aggregate_layer_transient(total)
        previous_stats = dict(getattr(
            self, "_glm53_layer_stationary_stats", {}) or {})
        self._glm53_layer_stationary_stats = {
            "memory_samples": int(previous_stats.get(
                "memory_samples", 0)) + memory_samples,
            "peak_metal_bytes": max(
                int(previous_stats.get("peak_metal_bytes", 0)),
                memory_peak),
            **{
                f"{phase}_active_peak_bytes": max(
                    int(previous_stats.get(
                        f"{phase}_active_peak_bytes", 0)), value)
                for phase, value in memory_phase_active.items()
            },
            "tile_width": int(tile_width),
            "positions": max(
                int(previous_stats.get("positions", 0)), total),
            "sweep_positions": int(previous_stats.get(
                "sweep_positions", 0)) + total,
            "sweeps": int(previous_stats.get("sweeps", 0)) + 1,
            "transient_reservation_calls": int(previous_stats.get(
                "transient_reservation_calls", 0)
                ) + transient_reservation_calls,
            "transient_reservation_bytes": int(previous_stats.get(
                "transient_reservation_bytes", 0)
                ) + transient_reservation_bytes,
            "transient_reservation_margin_bytes": int(previous_stats.get(
                "transient_reservation_margin_bytes", 0)
                ) + transient_reservation_margin_bytes,
            "transient_reservation_s": float(previous_stats.get(
                "transient_reservation_s", 0.0)
                ) + transient_reservation_s,
            "transient_reservation_first_margin_calls": int(
                previous_stats.get(
                    "transient_reservation_first_margin_calls", 0)
                ) + transient_reservation_first_margin_calls,
            "transient_reservation_recurring_calls": int(previous_stats.get(
                "transient_reservation_recurring_calls", 0)
                ) + transient_reservation_recurring_calls,
            "host_spool": int(host_spool),
            "host_spool_h2d_bytes": int(previous_stats.get(
                "host_spool_h2d_bytes", 0)) + spool_h2d_bytes,
            "host_spool_d2h_bytes": int(previous_stats.get(
                "host_spool_d2h_bytes", 0)) + spool_d2h_bytes,
            "host_spool_copy_s": float(previous_stats.get(
                "host_spool_copy_s", 0.0)) + spool_copy_s,
            "host_spool_peak_host_bytes": max(
                int(previous_stats.get(
                    "host_spool_peak_host_bytes", 0)),
                spool_peak_host_bytes),
        }
        return x

    def _layer_stationary_deepseek_v4_sweep(
            self, x: mx.array, kv, offset: int, tile_width: int,
            on_progress=None) -> mx.array:
        """Layer-major prefill for DeepSeek V4.

        Chunk-major prefill re-sweeps all 43 layers for every chunk, so a
        51,220-token harness prompt at chunk 384 costs 134 full passes over
        the model. Layer-major fetches each layer exactly once regardless of
        prompt length, which is the difference between correct and usable.

        Same split every other layer-stationary runner uses: attention runs
        per TILE, because the window ring and the compressor recurrence must
        see positions in causal order, while the MoE half runs ONCE per layer
        over every position. Routing is a function of its input alone, so
        evaluating it once on all positions rather than once per chunk on a
        subset is the same function on the union of its arguments.

        The per-tile position must be published on the cache before each
        attention call: _deepseek_v4_attention reads kv.dsv4_sweep_pos rather
        than kv.offset, since this attention owns its own ring and never calls
        kv.update(). Setting it once per sweep -- correct for chunk-major --
        would run every tile at the first tile's position.
        """
        from .deepseek_v4 import (deepseek_v4_attention_residual, hc_post,
                                  hc_pre)

        if tile_width <= 0:
            raise ValueError("tile_width must be positive")
        n = self.cfg.num_hidden_layers
        total = int(x.shape[1])
        # Size the reservation by the widest SINGLE step, not by the prompt.
        # Nothing here ever processes `total` positions at once: attention runs
        # per tile_width tile and the MoE per dsv4_ffn_tile_width group. Sizing
        # by total measured a whole LAYER -- carrier included -- and then
        # reserved that before every tile, so a 32,020-token prompt was refused
        # for 8.16GB it never intended to allocate in one go.
        probe_positions = min(
            total, max(tile_width, self.rc.dsv4_ffn_tile_width))
        (self._layer_transient,
         self._layer_transient_margin) = _layer_transient_for_positions(
             probe_positions,
             getattr(self, "_prefill_layer_transient_by_positions", {}
                     ).get(probe_positions, 0),
             getattr(self, "_decode_layer_transient", 0))
        profiler = self._request_profiler
        if profiler is not None:
            profiler.begin_sweep(total, path="layer_stationary_deepseek_v4")

        if offset == 0:
            kv.dsv4_pos = 0
        # The carrier is held as the LIST of tiles it is built from and is
        # never concatenated. It is positions x hc_mult x dim -- 32KB per
        # position, 1.68GB at 51K tokens -- and concatenating made it live
        # twice at the boundary, which killed a 51,220-token run outright
        # rather than refusing it. Both halves already work per tile, so the
        # single tensor was only ever an intermediate.
        # The carrier is [positions, hc_mult, dim]. A 51K census found it held
        # as float32 -- 3.347GB across 134 tiles, the single largest live
        # allocation -- because _packed_matmul returns float32, so q, the
        # attention output and the projection are all float32 and hc_post
        # inherits it via type_as. The released model runs that chain in its
        # own dtype, so narrowing the carrier both halves it and moves toward
        # released behaviour rather than away.
        spans = []
        pos = 0
        while pos < total:
            spans.append((pos, min(pos + tile_width, total)))
            pos = spans[-1][1]
        # Checkpoint boundary: the largest tile-aligned stride multiple
        # strictly inside this sweep. Captured DURING the single pass -- each
        # layer's ring/compressor state at position B is available the moment
        # that layer's tile loop crosses B, so collecting one entry per layer
        # yields the complete state at B without a second sweep.
        stride = self.rc.dsv4_prefix_checkpoint_stride
        checkpoint_at = 0
        if stride > 0 and offset == 0:
            aligned = (total - 1) // tile_width * tile_width
            checkpoint_at = min(aligned, (total - 1) // stride * stride)
            checkpoint_at = checkpoint_at // tile_width * tile_width
        checkpoint = {} if checkpoint_at else None

        expanded = mx.broadcast_to(
            x[:, :, None, :],
            (x.shape[0], total, self.cfg.hc_mult, x.shape[2]))
        tiles = [expanded[:, a:b] for a, b in spans]
        del expanded

        for i in range(n):
            self._select_layer_transient(probe_positions, i)
            if self.prefetcher:
                for j in range(i + 1, min(i + 1 + self.rc.prefetch_depth, n)):
                    self.prefetcher.schedule(self._layer_key(j),
                                             self._layer_names(j))
            cache_before = (profiler.cache_snapshot(self.cache)
                            if profiler is not None else None)
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
            if self._dsv4_packed_trunk:
                w = self._materialize_packed_trunk(w)
            weight_wait_s = time.perf_counter() - t0
            self.timer.add("weights_wait", weight_wait_s)

            prefix = f"model.layers.{i}"
            hc = {name: w[f"{prefix}.hc_{name}"]
                  for name in ("attn_fn", "attn_scale", "attn_base",
                               "ffn_fn", "ffn_scale", "ffn_base")}
            norms = {"attn": w[f"{prefix}.attn_norm.weight"],
                     "ffn": w[f"{prefix}.ffn_norm.weight"]}
            common = dict(hc_mult=self.cfg.hc_mult,
                          norm_eps=self.cfg.rms_norm_eps,
                          sinkhorn_iters=self.cfg.hc_sinkhorn_iters,
                          hc_eps=self.cfg.hc_eps)

            active_before = mx.get_active_memory()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            for index, (start, end) in enumerate(spans):
                if self.governor is not None and self._layer_transient:
                    self.governor.reserve(
                        self._layer_transient,
                        margin=self._layer_transient_margin)
                here = offset + start
                kv.dsv4_sweep_pos = here
                tile = deepseek_v4_attention_residual(
                    tiles[index], hc, norms,
                    lambda t, _h=here: self._deepseek_v4_attention(
                        t, w, prefix, i, kv, _h),
                    **common)
                if self._dsv4_carrier_dtype is not None:
                    tile = tile.astype(self._dsv4_carrier_dtype)
                mx.eval(tile)
                # Replace in place so the input tile is released now rather
                # than at the end of the layer.
                tiles[index] = tile
                if checkpoint is not None and end == checkpoint_at:
                    from .deepseek_v4 import CompressorState

                    state = (getattr(kv, "dsv4_cstate", None) or {}).get(i)
                    clone = None
                    if state is not None:
                        clone = CompressorState.__new__(CompressorState)
                        clone.__dict__.update(state.__dict__)
                    checkpoint[i] = {
                        "ring": (getattr(kv, "dsv4_rings", None) or {}).get(i),
                        "store": (getattr(kv, "dsv4_compressed", None)
                                  or {}).get(i),
                        "cstate": clone,
                    }

            # The MoE half is tiled too, but for MEMORY, not for weights.
            # hc_pre widens to float32, so one 51,220-position carrier is
            # 3.4GB and several are live at once -- a single call asked for
            # 13.4GB and exceeded Metal's maximum buffer size. Tiling caps
            # that while the expert pages stay resident ACROSS the tiles of
            # one layer, so they are still read once per layer: a whole
            # layer's routed set is 256 packed experts, about 3.4GB, which
            # fits the weight cache. Retention is restored afterwards so the
            # ordinary bounded-lifetime behaviour returns for decode.
            # The MoE runs ONCE per layer over every position, so each of
            # the layer's routed experts is read exactly once.
            #
            # The earlier once-per-layer attempt asked for 13.4GB and was
            # refused, which sent this down a per-group path -- but the size
            # was never the MoE. hc_pre widens the CARRIER, which is
            # [positions, hc_mult, dim]; the MoE's own input is the reduced
            # [positions, dim], four times smaller and bf16. Splitting the
            # hyper-connection wrapper (per tile) from the expert compute
            # (once) keeps both bounded: at 32K the reduced tensor is 262MB
            # where the carrier is 1.05GB.
            #
            # This matters because per-group re-fetching dominated read
            # traffic at length: 1,186.5GB at 32K against a layer's routed set
            # of ~3.4GB, i.e. every expert pulled roughly eight times.
            reduced_parts, posts, combs = [], [], []
            for index, tile in enumerate(tiles):
                r, post, comb = hc_pre(
                    tile, hc["ffn_fn"], hc["ffn_scale"], hc["ffn_base"],
                    hc_mult=self.cfg.hc_mult, norm_eps=self.cfg.rms_norm_eps,
                    sinkhorn_iters=self.cfg.hc_sinkhorn_iters,
                    eps=self.cfg.hc_eps)
                reduced_parts.append(
                    mx.fast.rms_norm(r, norms["ffn"], self.cfg.rms_norm_eps))
                posts.append(post)
                combs.append(comb)
            hidden = (reduced_parts[0] if len(reduced_parts) == 1
                      else mx.concatenate(reduced_parts, axis=1))
            del reduced_parts
            mx.eval(hidden)

            if self.governor is not None and self._layer_transient:
                self.governor.reserve(self._layer_transient,
                                      margin=self._layer_transient_margin)
            step_before = mx.get_active_memory()
            mx.reset_peak_memory()
            moe_out = self._deepseek_v4_ffn(hidden, w, prefix, i)
            mx.eval(moe_out)
            step_peak = _resident_adjusted_transient(
                step_before, mx.get_active_memory(), mx.get_peak_memory())
            del hidden

            for index, (start_pos_, end_pos_) in enumerate(spans):
                merged = hc_post(
                    moe_out[:, start_pos_:end_pos_], tiles[index],
                    posts[index], combs[index])
                if self._dsv4_carrier_dtype is not None:
                    merged = merged.astype(self._dsv4_carrier_dtype)
                tiles[index] = merged
                mx.eval(tiles[index])
            del moe_out, posts, combs

            compute_s = time.perf_counter() - t0
            self.timer.add("layer_compute", compute_s)
            if profiler is not None:
                profiler.record_layer(
                    i, positions=total, weight_wait_s=weight_wait_s,
                    compute_s=compute_s, cache_before=cache_before,
                    cache_after=profiler.cache_snapshot(self.cache),
                    layer_type=self._profile_layer_type(i))
            if on_progress is not None:
                on_progress({"phase": "prefill_layer",
                             "completed_layers": i + 1, "total_layers": n,
                             "total_tokens": total, "cache_source": "cold"})
            # Record what ONE step costs, keyed by that step's own width, so
            # the reservation stays honest however long the prompt is.
            self._record_layer_transient(probe_positions, i, step_peak)
            self._note_true_peak()
            del w

        kv.dsv4_pos = offset + total
        if checkpoint:
            self._dsv4_checkpoint = (checkpoint_at, checkpoint)
        from .deepseek_v4 import hc_head

        # Reduce per tile before joining: hc_head collapses the hc_mult axis,
        # so the concatenated result is a quarter of the carrier's size.
        for index in range(len(tiles)):
            tiles[index] = hc_head(
                tiles[index], self._hc_head_fn, self._hc_head_scale,
                self._hc_head_base, norm_eps=self.cfg.rms_norm_eps,
                eps=self.cfg.hc_eps)
            mx.eval(tiles[index])
        reduced = (tiles[0] if len(tiles) == 1
                   else mx.concatenate(tiles, axis=1))
        del tiles
        mx.eval(reduced)
        return reduced

    def _layer_stationary_kimi_linear_sweep(
            self, x: mx.array, kv, offset: int, tile_width: int,
            on_progress=None) -> mx.array:
        """F35-prep (2026-07-24): layer-major prefill for Kimi Linear, the
        MoE analogue of F94's dense-only `_layer_stationary_qwen35_sweep`.

        Quantified motivation (docs/future_lossless_techniques.md F92,
        2026-07-24 update): a real per-layer routed-expert footprint check
        against a real 291-token prompt found each MoE layer's TRUE unique-
        expert union across the whole prefill is only ~90-117/256, but the
        chunk-major loop's measured average weight-cache misses per layer
        was ~194 -- roughly double, the signature of chunk-major re-routing
        (and re-fetching) overlapping-but-not-identical expert sets once
        per chunk instead of once per layer.

        This fixes it by construction, not by any new "union of experts
        across chunks" bookkeeping: `_kimi_linear_attention_residual` still
        runs per TILE (attention/KDA state must still see tiles in causal
        order -- unchanged from chunk-major), but
        `_kimi_linear_mlp_residual` (routing + expert fetch + combination)
        now runs exactly ONCE per layer, on the FULL tiled-together
        attention output for every position in `x` -- there is no
        per-chunk MoE call left to redundantly re-route/re-fetch.

        `_layer_names(i)` already correctly excludes routed-expert tensor
        names for MoE models (`self.cfg.num_experts` gate, confirmed by
        reading the function directly) -- the SAME per-layer weight-fetch
        mechanism `_layer_stationary_qwen35_sweep` already uses is reused
        unmodified; the only new code is the attention/MLP call-site split
        itself (in runtime/kimi_linear.py) and this sweep's loop shape.

        Correctness argument mirrors `_layer_stationary_qwen35_sweep`'s own
        exactly: each layer's KDA/MLA state depends only on that layer's
        own sequential inputs and its own prior state, never on another
        layer's loop position, so reordering (layer, tile) -> outer layer,
        inner tile changes nothing about state evolution. Routing itself
        is stateless per call (a function of `h` alone) -- computing it
        once over all positions instead of once per chunk over a subset of
        positions is the SAME function evaluated on a UNION of its
        arguments, not a different function. Proven directly (not just
        argued) in tests/test_f35_kimi_linear_layer_stationary_oracle.py.
        """
        from .kimi_linear import (
            _kimi_linear_attention_residual, _kimi_linear_mlp_residual)

        if tile_width <= 0:
            raise ValueError("tile_width must be positive")
        n = self.cfg.num_hidden_layers
        total = int(x.shape[1])
        (self._layer_transient,
         self._layer_transient_margin) = _layer_transient_for_positions(
             total,
             getattr(
                 self, "_prefill_layer_transient_by_positions", {}
             ).get(total, 0),
             getattr(self, "_decode_layer_transient", 0))
        profiler = self._request_profiler
        if profiler is not None:
            profiler.begin_sweep(total, path="layer_stationary_kimi_linear")
        for i in range(n):
            self._select_layer_transient(total, i)
            if self.prefetcher:
                for j in range(i + 1, min(i + 1 + self.rc.prefetch_depth, n)):
                    self.prefetcher.schedule(self._layer_key(j), self._layer_names(j))

            cache_before = (
                profiler.cache_snapshot(self.cache)
                if profiler is not None else None)
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
            if self._dsv4_packed_trunk:
                w = self._materialize_packed_trunk(w)
            weight_wait_s = time.perf_counter() - t0
            self.timer.add("weights_wait", weight_wait_s)

            active_before = mx.get_active_memory()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            tiles = []
            pos = 0
            while pos < total:
                end = min(pos + tile_width, total)
                if self.governor is not None and self._layer_transient:
                    self.governor.reserve(
                        self._layer_transient,
                        margin=self._layer_transient_margin)
                xt = x[:, pos:end, :]
                attention_t0 = time.perf_counter()
                yt = _kimi_linear_attention_residual(
                    xt, w, f"model.layers.{i}", self.cfg, kv, i, offset + pos,
                    mlp_last_only=False,
                    native_fused_decode=self.rc.native_fused_deltanet_decode)
                mx.eval(yt)
                if profiler is not None and profiler.sync_substeps:
                    profiler.record_substep(
                        "attention", i,
                        time.perf_counter() - attention_t0,
                        positions=end - pos)
                tiles.append(yt)
                pos = end
            x_after_attn = tiles[0] if len(tiles) == 1 else mx.concatenate(tiles, axis=1)
            mlp_t0 = time.perf_counter()
            x = _kimi_linear_mlp_residual(
                x_after_attn, w, f"model.layers.{i}", self.cfg, i,
                self._get_experts,
                iter_expert_batches=self._iter_expert_batches,
                profile=profiler)
            mx.eval(x)
            if profiler is not None and profiler.sync_substeps:
                profiler.record_substep(
                    "mlp", i, time.perf_counter() - mlp_t0,
                    positions=total)
            compute_s = time.perf_counter() - t0
            self.timer.add("layer_compute", compute_s)
            if profiler is not None:
                profiler.record_layer(
                    i, positions=total, weight_wait_s=weight_wait_s,
                    compute_s=compute_s, cache_before=cache_before,
                    cache_after=profiler.cache_snapshot(self.cache),
                    layer_type=self._profile_layer_type(i),
                )
            if on_progress is not None:
                on_progress({
                    "phase": "prefill_layer",
                    "completed_layers": i + 1,
                    "total_layers": n,
                    "total_tokens": total,
                    "cache_source": "cold",
                })
            self._record_layer_transient(
                total, i,
                _resident_adjusted_transient(
                    active_before, mx.get_active_memory(),
                    mx.get_peak_memory()))
            self._note_true_peak()
            del w
        self._restore_aggregate_layer_transient(total)
        return x

    def _layer_stationary_glm5_next_sweep(
            self, x: mx.array, kv, offset: int, tile_width: int,
            on_progress=None, tap_layers=None, *,
            incremental_expanded_mla: bool = True) -> mx.array:
        """Layer-major bounded prefill for GLM-5.3's mHC/KDA/DSA stack.

        The released block is separable by position around its two stateful
        attention families: mHC maps and the MLP/router are row functions,
        while KDA and cached MLA consume rows causally.  Keep the carrier as
        bounded tiles, advance attention in chronological order, then evaluate
        the layer's MLP once over the concatenated reduced rows.  Consequently
        a routed expert is fetched at most once for this layer instead of once
        per prompt tile, without changing router inputs, expert accumulation
        order or target weights.  The DSA projection uses one bounded latent
        tile at a time; that changes GEMM shape relative to ordinary compressed
        decode and is therefore E-class unless a caller separately proves the
        required released-state/token equivalence for its request corpus.
        """
        from .deepseek_v4 import (deepseek_v4_attention_residual, hc_post,
                                  hc_pre)
        from .glm5_next import (
            glm5_next_mla_attention,
            glm5_next_mlp_layer_stationary_tiles,
        )
        from .kimi_linear import _kda_attention
        import numpy as np

        if tile_width <= 0:
            raise ValueError("tile_width must be positive")
        if self.cfg.model_type != "glm5_next":
            raise ValueError("GLM-5.3 layer-stationary path needs glm5_next")
        total = int(x.shape[1])
        if total <= 0:
            return x
        spans = [
            (start, min(start + tile_width, total))
            for start in range(0, total, tile_width)
        ]
        host_spool = bool(self.rc.glm53_layer_stationary_host_spool)
        spool_h2d_bytes = 0
        spool_d2h_bytes = 0
        spool_copy_s = 0.0
        spool_peak_host_bytes = 0

        def host_bits(value: mx.array) -> np.ndarray:
            """Copy one released BF16 tile without a numeric conversion."""
            nonlocal spool_d2h_bytes, spool_copy_s
            if value.dtype != mx.bfloat16:
                raise TypeError(
                    "GLM-5.3 host spool requires BF16 carrier tiles, "
                    f"got {value.dtype}")
            started = time.perf_counter()
            mx.eval(value)
            result = np.array(
                np.asarray(value.view(mx.uint16)), dtype=np.uint16, copy=True)
            spool_d2h_bytes += int(result.nbytes)
            spool_copy_s += time.perf_counter() - started
            return result

        def metal_bits(value: np.ndarray) -> mx.array:
            """Restore one raw BF16 host tile at its original shape."""
            nonlocal spool_h2d_bytes, spool_copy_s
            started = time.perf_counter()
            result = mx.array(value, dtype=mx.uint16).view(mx.bfloat16)
            mx.eval(result)
            spool_h2d_bytes += int(value.nbytes)
            spool_copy_s += time.perf_counter() - started
            return result

        expanded = mx.broadcast_to(
            x[:, :, None, :],
            (x.shape[0], total, self.cfg.hc_mult, x.shape[2]))
        if host_spool:
            tiles = [
                host_bits(expanded[:, start:end]) for start, end in spans
            ]
            spool_peak_host_bytes = sum(int(tile.nbytes) for tile in tiles)
        else:
            tiles = [expanded[:, start:end] for start, end in spans]
        collector = (
            self._dspark_tap_collector if tap_layers is None else None)
        if collector is not None:
            tap_layers = collector.tap_layers
        tapset = set(tap_layers) if tap_layers is not None else set()
        del expanded, x
        if host_spool:
            mx.clear_cache()
        # Ordinary prefill keeps each projection only for the currently active
        # DSA layer; the durable endpoint remains the released compact latent.
        # This removes quadratic re-projection, but its bounded GEMM shape can
        # round differently from ordinary growing-prefix compressed decode.
        # The target-exact serial verifier therefore disables this cache and
        # reprojects the same growing latent prefix once per position, matching
        # canonical one-token decode while still loading each layer only once.
        kv._glm53_expanded_prefill = (
            _glm53_expanded_prefill_cache(kv)
            if incremental_expanded_mla else None)

        probe_positions = min(total, tile_width)
        (self._layer_transient,
         self._layer_transient_margin) = _layer_transient_for_positions(
             probe_positions,
             getattr(self, "_prefill_layer_transient_by_positions", {}).get(
                 probe_positions, 0),
             getattr(self, "_decode_layer_transient", 0))
        profiler = self._request_profiler
        metal_limit_bytes = max(0, int(self.rc.metal_limit_mb)) * 1_000_000
        memory_samples = 0
        memory_peak = 0
        memory_phase_active = {
            "initial_carrier": 0,
            "attention": 0,
            "ffn_hc_pre": 0,
            "mlp": 0,
            "ffn_hc_post": 0,
        }
        fused_calls_before = int(getattr(
            kv, "_glm53_sparse_fused_calls", 0))
        fused_positions_before = int(getattr(
            kv, "_glm53_sparse_fused_positions", 0))
        fused_rows_before = int(getattr(
            kv, "_glm53_sparse_fused_selected_rows", 0))
        layer_weight_wait_s = 0.0
        attention_residual_s = 0.0
        kda_attention_s = 0.0
        mla_attention_s = 0.0
        ffn_hc_pre_s = 0.0
        mlp_s = 0.0
        ffn_hc_post_s = 0.0
        coalesced_stats = {
            "layers": 0,
            "input_positions": 0,
            "route_assignments": 0,
            "unique_experts": 0,
            "max_unique_experts": 0,
            "max_expert_routes": 0,
            "gemm_calls": 0,
            "gemm_input_positions": 0,
            "gemm_full_chunks": 0,
            "max_positions": 0,
            "split_experts": 0,
        }

        def note_memory(
                phase: str, layer: int, completed_tokens: int = 0,
                *, publish: bool = False) -> None:
            """Content-blind long-prefill attribution plus a synchronous cap."""
            nonlocal memory_samples, memory_peak
            active = int(mx.get_active_memory())
            peak = int(mx.get_peak_memory())
            observed = max(active, peak)
            if phase.startswith("attention"):
                phase_group = "attention"
            elif phase.startswith("ffn_hc_pre"):
                phase_group = "ffn_hc_pre"
            elif phase.startswith("ffn_hc_post"):
                phase_group = "ffn_hc_post"
            elif phase in (
                    "dense_mlp_tile", "router_tile", "shared_mlp_tile",
                    "routed_expert_batch"):
                phase_group = "mlp"
            else:
                phase_group = "initial_carrier"
            memory_phase_active[phase_group] = max(
                memory_phase_active[phase_group], active)
            memory_samples += 1
            memory_peak = max(memory_peak, observed)
            self._note_true_peak()
            if publish and on_progress is not None:
                on_progress({
                    "phase": "prefill_layer",
                    "diagnostic": "glm53_layer_stationary",
                    "subphase": phase,
                    "layer": int(layer),
                    "completed_layers": max(0, int(layer)),
                    "total_layers": self.cfg.num_hidden_layers,
                    "completed_tokens": int(completed_tokens),
                    "total_tokens": total,
                    "active_metal_bytes": active,
                    "peak_metal_bytes": peak,
                    "host_spool_bytes": int(spool_peak_host_bytes),
                    "metal_limit_bytes": metal_limit_bytes,
                    "cache_source": "cold",
                })
            if metal_limit_bytes and observed > metal_limit_bytes:
                raise MemoryError(
                    "GLM-5.3 layer-stationary prefill crossed its hard Metal "
                    f"cap: phase={phase} layer={layer} "
                    f"tokens={completed_tokens} observed={observed} "
                    f"limit={metal_limit_bytes}")

        note_memory("initial_carrier", 0, total, publish=True)
        if profiler is not None:
            profiler.begin_sweep(
                total, path="layer_stationary_glm5_next")

        for layer in range(self.cfg.num_hidden_layers):
            self._select_layer_transient(probe_positions, layer)
            if self.prefetcher:
                for nxt in range(
                        layer + 1,
                        min(layer + 1 + self.rc.prefetch_depth,
                            self.cfg.num_hidden_layers)):
                    self.prefetcher.schedule(
                        self._layer_key(nxt), self._layer_names(nxt))
            cache_before = (
                profiler.cache_snapshot(self.cache)
                if profiler is not None else None)
            wait_started = time.perf_counter()
            layer_key = self._layer_key(layer)
            layer_names = self._layer_names(layer)
            if not self.cache.contains(layer_key):
                incoming_page = self._layer_fetch_bytes_estimate(layer)
                if incoming_page:
                    self.cache.prepare_for(incoming_page)
                    if self.governor is not None:
                        self.governor.reserve(
                            incoming_page, reason="glm53-layer-page")
            w = self.cache.get(layer_key, layer_names)
            weight_wait_s = time.perf_counter() - wait_started
            layer_weight_wait_s += weight_wait_s
            self.timer.add("weights_wait", weight_wait_s)

            prefix = f"model.layers.{layer}"
            hc = {
                name: w[f"{prefix}.hc_{name}"]
                for name in (
                    "attn_fn", "attn_scale", "attn_base",
                    "ffn_fn", "ffn_scale", "ffn_base")
            }
            norms = {
                "attn": w[f"{prefix}.input_layernorm.weight"],
                "ffn": w[f"{prefix}.post_attention_layernorm.weight"],
            }
            common = dict(
                hc_mult=self.cfg.hc_mult,
                norm_eps=self.cfg.rms_norm_eps,
                sinkhorn_iters=self.cfg.hc_sinkhorn_iters,
                hc_eps=self.cfg.hc_eps,
            )
            active_before = mx.get_active_memory()
            mx.reset_peak_memory()
            compute_started = time.perf_counter()

            for index, (start, end) in enumerate(spans):
                if self.governor is not None and self._layer_transient:
                    self.governor.reserve(
                        self._layer_transient,
                        margin=self._layer_transient_margin,
                        reason="glm53-attention-transient")
                here = offset + start

                def attention(hidden, *, _here=here):
                    if layer in self.cfg.kda_layers:
                        return _kda_attention(
                            hidden, w, prefix, self.cfg,
                            getattr(kv, "kda_cache", None), layer,
                            native_fused_decode=(
                                self.rc.native_fused_deltanet_decode),
                            native_fused_prefill=(
                                self.rc.glm53_native_fused_kda_prefill),
                            compiled_prefill=(
                                self.rc.glm53_compiled_kda_prefill),
                            compiled_prefill_segment=(
                                self.rc.glm53_compiled_kda_segment),
                            released_output_dtype=True,
                            profile=profiler,
                        )
                    if layer in self.cfg.full_attn_layers:
                        return glm5_next_mla_attention(
                            hidden, w, prefix, self.cfg, kv, layer, _here)
                    raise ValueError(
                        f"GLM-5.3 layer {layer} has no attention type")

                attention_tile_started = time.perf_counter()
                source_tile = (
                    metal_bits(tiles[index]) if host_spool else tiles[index])
                tile = deepseek_v4_attention_residual(
                    source_tile, hc, norms, attention, **common)
                tile = tile.astype(mx.bfloat16)
                mx.eval(tile)
                attention_tile_s = (
                    time.perf_counter() - attention_tile_started)
                attention_residual_s += attention_tile_s
                if layer in self.cfg.kda_layers:
                    kda_attention_s += attention_tile_s
                else:
                    mla_attention_s += attention_tile_s
                tiles[index] = host_bits(tile) if host_spool else tile
                if host_spool:
                    source_tile = tile = None
                    mx.clear_cache()
                note_memory(
                    "attention_tile", layer, end,
                    publish=(index == 0 or end == total
                             or index % 64 == 0))

            if layer in self.cfg.full_attn_layers:
                _glm53_release_expanded_prefill_layer(kv, layer)
                mx.clear_cache()

            # Bound the float32 mHC mapping by tile and retain each normalized
            # row group separately. The MLP helper preserves those exact GEMM
            # shapes while sharing each routed expert page across the tiles.
            hidden_tiles, posts, combs = [], [], []
            for index, (tile, (_start, end)) in enumerate(zip(tiles, spans)):
                hc_pre_started = time.perf_counter()
                source_tile = metal_bits(tile) if host_spool else tile
                reduced, post, comb = hc_pre(
                    source_tile, hc["ffn_fn"], hc["ffn_scale"], hc["ffn_base"],
                    hc_mult=self.cfg.hc_mult,
                    norm_eps=self.cfg.rms_norm_eps,
                    sinkhorn_iters=self.cfg.hc_sinkhorn_iters,
                    eps=self.cfg.hc_eps)
                hidden = mx.fast.rms_norm(
                    reduced, norms["ffn"], self.cfg.rms_norm_eps)
                # Long GLM prompts contain thousands of small tiles. Passing
                # every lazy hc_pre output to one giant mx.eval() retains the
                # float32 flattened mHC carrier and 20-iteration Sinkhorn graph
                # for the whole context at once (46,849 positions measured a
                # 14.0-GB active Metal peak). Materialize and sever each tile's
                # graph here; arithmetic and per-tile operator order are
                # unchanged, while live scratch is O(tile_width), not O(S).
                mx.eval(hidden, post, comb)
                ffn_hc_pre_s += time.perf_counter() - hc_pre_started
                hidden_tiles.append(hidden)
                posts.append(post)
                combs.append(comb)
                if host_spool:
                    source_tile = reduced = None
                    mx.clear_cache()
                note_memory(
                    "ffn_hc_pre_tile", layer, end,
                    publish=(index == 0 or end == total
                             or index % 64 == 0))

            if self.governor is not None and self._layer_transient:
                self.governor.reserve(
                    self._layer_transient,
                    margin=self._layer_transient_margin,
                    reason="glm53-mlp-transient")
            step_before = mx.get_active_memory()
            mx.reset_peak_memory()
            mlp_started = time.perf_counter()
            mlp_tiles = glm5_next_mlp_layer_stationary_tiles(
                hidden_tiles, w, prefix, self.cfg, layer, self._get_experts,
                iter_expert_batches=self._iter_expert_batches,
                profile=profiler,
                coalesce_expert_positions=(
                    self.rc.glm53_coalesced_expert_positions),
                coalesced_expert_max_positions=(
                    self.rc.glm53_coalesced_expert_max_positions),
                coalesced_stats=coalesced_stats,
                memory_guard=lambda phase, _layer=layer: note_memory(
                    phase, _layer, total, publish=False))
            mlp_s += time.perf_counter() - mlp_started
            step_peak = _resident_adjusted_transient(
                step_before, mx.get_active_memory(), mx.get_peak_memory())
            del hidden_tiles

            for index, (start, end) in enumerate(spans):
                hc_post_started = time.perf_counter()
                source_tile = (
                    metal_bits(tiles[index])
                    if host_spool else tiles[index])
                merged = hc_post(
                    mlp_tiles[index], source_tile,
                    posts[index], combs[index]).astype(mx.bfloat16)
                mx.eval(merged)
                ffn_hc_post_s += time.perf_counter() - hc_post_started
                tiles[index] = host_bits(merged) if host_spool else merged
                if host_spool:
                    source_tile = merged = None
                    mx.clear_cache()
                note_memory(
                    "ffn_hc_post_tile", layer, end,
                    publish=(index == 0 or end == total
                             or index % 64 == 0))
            del mlp_tiles, posts, combs

            if layer in tapset:
                tapped_tiles = []
                for tile in tiles:
                    source_tile = metal_bits(tile) if host_spool else tile
                    tapped = mx.mean(
                        source_tile, axis=2).astype(mx.bfloat16)
                    mx.eval(tapped)
                    tapped_tiles.append(tapped)
                    source_tile = None
                self._tap_hidden[layer] = (
                    tapped_tiles[0] if len(tapped_tiles) == 1
                    else mx.concatenate(tapped_tiles, axis=1))
                mx.eval(self._tap_hidden[layer])
                if collector is not None:
                    collector.observe(
                        layer, self._tap_hidden[layer],
                        position_start=offset)

            compute_s = time.perf_counter() - compute_started
            self.timer.add("layer_compute", compute_s)
            if profiler is not None:
                profiler.record_layer(
                    layer, positions=total,
                    weight_wait_s=weight_wait_s, compute_s=compute_s,
                    cache_before=cache_before,
                    cache_after=profiler.cache_snapshot(self.cache),
                    layer_type=self._profile_layer_type(layer))
            if on_progress is not None:
                on_progress({
                    "phase": "prefill_layer",
                    "completed_layers": layer + 1,
                    "total_layers": self.cfg.num_hidden_layers,
                    "total_tokens": total,
                    "cache_source": "cold",
                })
            self._record_layer_transient(probe_positions, layer, step_peak)
            self._note_true_peak()
            del w

        # The released hyper head is an unweighted mean. Collapse each tile
        # before joining so the temporary is one quarter of the carrier.
        for index, tile in enumerate(tiles):
            source_tile = metal_bits(tile) if host_spool else tile
            tiles[index] = mx.mean(
                source_tile, axis=2).astype(mx.bfloat16)
            mx.eval(tiles[index])
            source_tile = None
        result = (
            tiles[0] if len(tiles) == 1
            else mx.concatenate(tiles, axis=1))
        mx.eval(result)
        previous_stats = dict(getattr(
            self, "_glm53_layer_stationary_stats", {}) or {})
        self._glm53_layer_stationary_stats = {
            "memory_samples": int(previous_stats.get(
                "memory_samples", 0)) + memory_samples,
            "peak_metal_bytes": max(
                int(previous_stats.get("peak_metal_bytes", 0)),
                memory_peak),
            **{
                f"{phase}_active_peak_bytes": max(
                    int(previous_stats.get(
                        f"{phase}_active_peak_bytes", 0)), value)
                for phase, value in memory_phase_active.items()
            },
            "tile_width": int(tile_width),
            "positions": max(
                int(previous_stats.get("positions", 0)), total),
            "sweep_positions": int(previous_stats.get(
                "sweep_positions", 0)) + total,
            "sweeps": int(previous_stats.get("sweeps", 0)) + 1,
            "host_spool": int(host_spool),
            "host_spool_h2d_bytes": int(previous_stats.get(
                "host_spool_h2d_bytes", 0)) + spool_h2d_bytes,
            "host_spool_d2h_bytes": int(previous_stats.get(
                "host_spool_d2h_bytes", 0)) + spool_d2h_bytes,
            "host_spool_copy_s": float(previous_stats.get(
                "host_spool_copy_s", 0.0)) + spool_copy_s,
            "host_spool_peak_host_bytes": max(
                int(previous_stats.get("host_spool_peak_host_bytes", 0)),
                spool_peak_host_bytes),
            "sparse_fused_calls": int(previous_stats.get(
                "sparse_fused_calls", 0)) + int(getattr(
                    kv, "_glm53_sparse_fused_calls", 0)) - fused_calls_before,
            "sparse_fused_positions": int(previous_stats.get(
                "sparse_fused_positions", 0)) + int(getattr(
                    kv, "_glm53_sparse_fused_positions", 0)) - fused_positions_before,
            "sparse_fused_selected_rows": int(previous_stats.get(
                "sparse_fused_selected_rows", 0)) + int(getattr(
                    kv, "_glm53_sparse_fused_selected_rows", 0)) - fused_rows_before,
            "weight_wait_s": float(previous_stats.get(
                "weight_wait_s", 0.0)) + layer_weight_wait_s,
            "attention_residual_s": float(previous_stats.get(
                "attention_residual_s", 0.0)) + attention_residual_s,
            "kda_attention_s": float(previous_stats.get(
                "kda_attention_s", 0.0)) + kda_attention_s,
            "mla_attention_s": float(previous_stats.get(
                "mla_attention_s", 0.0)) + mla_attention_s,
            "ffn_hc_pre_s": float(previous_stats.get(
                "ffn_hc_pre_s", 0.0)) + ffn_hc_pre_s,
            "mlp_s": float(previous_stats.get(
                "mlp_s", 0.0)) + mlp_s,
            "ffn_hc_post_s": float(previous_stats.get(
                "ffn_hc_post_s", 0.0)) + ffn_hc_post_s,
            **{
                f"exact_expert_{metric}": int(previous_stats.get(
                    f"exact_expert_{metric}", 0)) + int(
                        coalesced_stats.get(
                            f"exact_expert_{metric}", 0))
                for metric in (
                    "layers", "tiles", "swiglu_calls", "rows",
                    "rows_1_calls", "rows_2_calls", "rows_3_4_calls",
                    "rows_5_8_calls", "rows_9_16_calls",
                    "rows_17_32_calls", "rows_33_plus_calls")
            },
            "exact_expert_max_rows": max(
                int(previous_stats.get("exact_expert_max_rows", 0)),
                int(coalesced_stats.get("exact_expert_max_rows", 0))),
            "coalesced_expert_gemm_calls": int(previous_stats.get(
                "coalesced_expert_gemm_calls", 0)) + int(
                    coalesced_stats["gemm_calls"]),
            "coalesced_expert_max_positions": max(
                int(previous_stats.get(
                    "coalesced_expert_max_positions", 0)),
                int(coalesced_stats["max_positions"])),
            "coalesced_expert_split_experts": int(previous_stats.get(
                "coalesced_expert_split_experts", 0)) + int(
                    coalesced_stats["split_experts"]),
            "coalesced_expert_layers": int(previous_stats.get(
                "coalesced_expert_layers", 0)) + int(
                    coalesced_stats["layers"]),
            "coalesced_expert_input_positions": int(previous_stats.get(
                "coalesced_expert_input_positions", 0)) + int(
                    coalesced_stats["input_positions"]),
            "coalesced_expert_route_assignments": int(previous_stats.get(
                "coalesced_expert_route_assignments", 0)) + int(
                    coalesced_stats["route_assignments"]),
            "coalesced_expert_unique_experts": int(previous_stats.get(
                "coalesced_expert_unique_experts", 0)) + int(
                    coalesced_stats["unique_experts"]),
            "coalesced_expert_max_unique_experts": max(
                int(previous_stats.get(
                    "coalesced_expert_max_unique_experts", 0)),
                int(coalesced_stats["max_unique_experts"])),
            "coalesced_expert_max_expert_routes": max(
                int(previous_stats.get(
                    "coalesced_expert_max_expert_routes", 0)),
                int(coalesced_stats["max_expert_routes"])),
            "coalesced_expert_gemm_input_positions": int(previous_stats.get(
                "coalesced_expert_gemm_input_positions", 0)) + int(
                    coalesced_stats["gemm_input_positions"]),
            "coalesced_expert_gemm_full_chunks": int(previous_stats.get(
                "coalesced_expert_gemm_full_chunks", 0)) + int(
                    coalesced_stats["gemm_full_chunks"]),
        }
        del kv._glm53_expanded_prefill
        self._restore_aggregate_layer_transient(total)
        return result

    def _layer_stationary_kimi_k3_sweep(
            self, x: mx.array, kv, offset: int, tile_width: int,
            on_progress=None) -> mx.array:
        """F128: AttnRes-aware layer-stationary prefill for Kimi K3 -- same
        motivation and per-layer weight-fetch mechanism as
        `_layer_stationary_kimi_linear_sweep` above (each layer's weights
        fetched exactly once for the whole prefill range, not once per
        chunk), generalized for AttnRes's extra `block_residual` state.

        `attn_res_wrap_layer` (runtime/kimi_linear.py) already accepts
        arbitrary `attn_fn`/`mlp_fn` closures that receive/return the FULL
        `(B, L, H)` tensor for whatever positions are passed in -- nothing
        about it assumes attention runs in one shot. This reuses that
        directly: `attn_fn` here tiles internally (attention must still see
        tiles in causal order, exactly like the plain kimi_linear sweep
        above), while `mlp_fn` runs once over the whole layer's positions
        as before. The AttnRes pre/post mixing itself
        (`_apply_attn_res`) is a per-position, row-independent operation
        with no causal-order requirement, so applying it ONCE over all of
        `x` (inside `attn_res_wrap_layer`, called once per LAYER here, not
        once per tile) is exactly equivalent to applying it per-tile --
        only the block-boundary snapshot/reset bookkeeping needs to happen
        at layer granularity, which calling `attn_res_wrap_layer` exactly
        once per outer-loop iteration already gives for free.
        """
        from .kimi_linear import (
            _kda_attention, _mla_attention, _kimi_dense_mlp_tiled,
            _kimi_moe_output,
            attn_res_wrap_layer, attn_res_wrap_layer_streamed,
            apply_output_attn_res,
            DiskBackedAttnResSnapshots)

        if tile_width <= 0:
            raise ValueError("tile_width must be positive")
        if (
            self.store.bf16_nf12_sidecar is not None
            and not self.store.bf16_nf12_uncached_reads
        ):
            return self._layer_stationary_kimi_k3_nf12_split_sweep(
                x, kv, offset, tile_width, on_progress=on_progress
            )
        n = self.cfg.num_hidden_layers
        total = int(x.shape[1])
        (self._layer_transient,
         self._layer_transient_margin) = _layer_transient_for_positions(
             total,
             getattr(
                 self, "_prefill_layer_transient_by_positions", {}
             ).get(total, 0),
             getattr(self, "_decode_layer_transient", 0))
        profiler = self._request_profiler
        if profiler is not None:
            profiler.begin_sweep(total, path="layer_stationary_kimi_k3")
        if self.rc.kimi_k3_attnres_spill_dir:
            block_residual = DiskBackedAttnResSnapshots(
                self.rc.kimi_k3_attnres_spill_dir,
                write_tile_rows=tile_width,
            )
        else:
            block_residual = (
                []
                if self.rc.kimi_k3_fused_attnres_tile_size
                else mx.zeros(
                    (x.shape[0] * total, 0, x.shape[2]), dtype=x.dtype
                )
            )
        for i in range(n):
            self._select_layer_transient(total, i)
            if self.prefetcher:
                for j in range(i + 1, min(i + 1 + self.rc.prefetch_depth, n)):
                    hint = self._layer_fetch_bytes_estimate(j)
                    self.prefetcher.schedule(
                        self._layer_key(j),
                        self._layer_names(j),
                        page_size_hint=hint or None,
                    )

            cache_before = (
                profiler.cache_snapshot(self.cache)
                if profiler is not None else None)
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
            if self._dsv4_packed_trunk:
                w = self._materialize_packed_trunk(w)
            weight_wait_s = time.perf_counter() - t0
            self.timer.add("weights_wait", weight_wait_s)

            active_before = mx.get_active_memory()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            prefix = f"model.layers.{i}"

            def attn_tile_fn(ht, start, end, i=i, prefix=prefix):
                if self.governor is not None and self._layer_transient:
                    reserve_margin = self._layer_transient_margin
                    if isinstance(
                        block_residual, DiskBackedAttnResSnapshots
                    ):
                        key = (
                            total,
                            self._transient_layer_signature(i),
                        )
                        observations = int(getattr(
                            self,
                            "_layer_transient_observation_counts",
                            {},
                        ).get(key, 0))
                        reserve_margin = (
                            _recurring_layer_transient_reserve_margin(
                                total, observations)
                        )
                    self.governor.reserve(
                        self._layer_transient,
                        margin=reserve_margin)
                attention_t0 = time.perf_counter()
                if i in self.cfg.full_attn_layers:
                    yt = _mla_attention(
                        ht, w, prefix, self.cfg, kv, i, offset + start)
                elif i in self.cfg.kda_layers:
                    kda_cache = getattr(kv, "kda_cache", None)
                    yt = _kda_attention(
                        ht, w, prefix, self.cfg, kda_cache, i,
                        native_fused_decode=(
                            self.rc.native_fused_deltanet_decode),
                        native_fused_prefill=(
                            self.rc.kimi_k3_native_fused_kda_prefill),
                        compiled_prefill=(
                            self.rc.kimi_k3_compiled_kda_prefill),
                        profile=profiler)
                else:
                    raise ValueError(
                        f"layer {i} is in neither cfg.full_attn_layers "
                        "nor cfg.kda_layers")
                mx.eval(yt)
                if profiler is not None and profiler.sync_substeps:
                    profiler.record_substep(
                        "attention", i,
                        time.perf_counter() - attention_t0,
                        positions=end - start)
                return yt

            def attn_fn(hidden_states, i=i, prefix=prefix):
                tiles = []
                pos = 0
                while pos < total:
                    end = min(pos + tile_width, total)
                    ht = hidden_states[:, pos:end, :]
                    yt = attn_tile_fn(ht, pos, end)
                    tiles.append(yt)
                    pos = end
                return tiles[0] if len(tiles) == 1 else mx.concatenate(tiles, axis=1)

            def mlp_fn(h2, i=i, prefix=prefix):
                mlp_t0 = time.perf_counter()
                if i < self.cfg.first_k_dense_replace:
                    out = _kimi_dense_mlp_tiled(
                        h2,
                        w,
                        f"{prefix}.mlp",
                        self.cfg,
                        self.rc.kimi_k3_dense_mlp_tile_size,
                    )
                else:
                    if self.rc.expert_batch_prefetch:
                        self._expert_shared_overlap_layers += 1
                    h2_owner = [h2]
                    del h2
                    streamed_moe = isinstance(
                        block_residual, DiskBackedAttnResSnapshots)
                    saved_transient = self._layer_transient
                    if streamed_moe:
                        # At expert-fetch time the streamed wrapper has
                        # already materialized the full MLP input, and the
                        # tiled latent-MoE path has evaluated its latent and
                        # shared branches.  mx.get_active_memory() therefore
                        # accounts for those live buffers.  Adding the whole
                        # historical layer transient again double-counted
                        # them and refused 8.16 GB projected despite a 7.19
                        # GB measured dense+MoE peak.  Reserve compact expert
                        # pages against the authoritative live sample; the
                        # ordinary/non-streamed paths retain their existing
                        # fail-closed transient reservation.
                        self._layer_transient = 0
                    try:
                        out = _kimi_moe_output(
                            h2_owner.pop(),
                            w, prefix, self.cfg, i, self._get_experts,
                            iter_expert_batches=self._iter_expert_batches,
                            profile=profiler,
                            overlap_shared_expert=(
                                self.rc.expert_batch_prefetch),
                            shared_tile_size=(
                                self.rc.kimi_k3_dense_mlp_tile_size
                                if streamed_moe else 0),
                        )
                    finally:
                        self._layer_transient = saved_transient
                mx.eval(out)
                if profiler is not None and profiler.sync_substeps:
                    profiler.record_substep(
                        "mlp", i, time.perf_counter() - mlp_t0, positions=total)
                return out

            if isinstance(block_residual, DiskBackedAttnResSnapshots):
                # Transfer the prior activation's sole owner into the streamed
                # wrapper so it can release that full-context buffer before
                # entering the much larger MoE lifetime.
                x_owner = [x]
                del x
                x, block_residual = attn_res_wrap_layer_streamed(
                    x_owner.pop(),
                    block_residual, w, prefix, self.cfg, i,
                    attn_tile_fn, mlp_fn,
                    tile_size=tile_width,
                    fused_tile_size=(
                        self.rc.kimi_k3_fused_attnres_tile_size))
            else:
                x, block_residual = attn_res_wrap_layer(
                    x, block_residual, w, prefix, self.cfg, i,
                    attn_fn, mlp_fn,
                    fused_tile_size=(
                        self.rc.kimi_k3_fused_attnres_tile_size))
            mx.eval(x)
            if (
                self.rc.kimi_k3_kda_spill_dir
                and i in self.cfg.kda_layers
                and getattr(kv, "kda_cache", None) is not None
                and kv.kda_cache.spill_layer(i)
            ):
                # The completed endpoint is now exact raw bytes on the
                # external tier. No later prefill layer consumes it; decode
                # reloads that layer lazily through the same cache interface.
                mx.clear_cache()
            if (
                self.rc.kimi_k3_mla_kv_spill_dir
                and i in self.cfg.full_attn_layers
                and getattr(kv, "latent_spill_enabled", False)
                and kv.spill_latent_layer(i)
            ):
                # Later prefill layers never attend through an earlier layer's
                # KV. Decode reloads this exact latent only when it reaches the
                # corresponding full-attention layer.
                mx.clear_cache()
            compute_s = time.perf_counter() - t0
            self.timer.add("layer_compute", compute_s)
            if profiler is not None:
                profiler.record_layer(
                    i, positions=total, weight_wait_s=weight_wait_s,
                    compute_s=compute_s, cache_before=cache_before,
                    cache_after=profiler.cache_snapshot(self.cache),
                    layer_type=self._profile_layer_type(i),
                )
            if on_progress is not None:
                on_progress({
                    "phase": "prefill_layer",
                    "completed_layers": i + 1,
                    "total_layers": n,
                    "total_tokens": total,
                    "cache_source": "cold",
                })
            end_active = mx.get_active_memory()
            peak_active = mx.get_peak_memory()
            measured_transient = _resident_adjusted_transient(
                active_before, end_active, peak_active)
            self._last_k3_transient_observation = {
                "layer": i,
                "signature": self._transient_layer_signature(i),
                "start_active_bytes": int(active_before),
                "end_active_bytes": int(end_active),
                "peak_active_bytes": int(peak_active),
                "measured_transient_bytes": int(measured_transient),
            }
            self._record_layer_transient(
                total, i, measured_transient)
            self._note_true_peak()
            del w
            if self.store.bf16_nf12_sidecar is not None:
                # F140: cache-budget eviction may precede the actual consumer
                # boundary. The exact decoded trunk and its mapped NF12 source
                # become disposable only after this layer's synchronized
                # compute, transient accounting, and local weight reference
                # have all completed. Explicitly drop the one-pass page now,
                # then let WeightStore retry Darwin UBC invalidation.
                self.cache.discard(layer_key, layer_names)
        self._restore_aggregate_layer_transient(total)
        x = apply_output_attn_res(
            x, {
                "model.output_attn_res_proj.weight": self._output_attn_res_proj_w,
                "model.output_attn_res_norm.weight": self._output_attn_res_norm_w,
            }, block_residual, self.cfg,
            fused_tile_size=(
                self.rc.kimi_k3_fused_attnres_tile_size))
        if isinstance(block_residual, DiskBackedAttnResSnapshots):
            self._last_k3_attnres_spill_stats = block_residual.stats()
            block_residual.close()
        return x

    def _layer_stationary_kimi_k3_nf12_split_sweep(
            self, x: mx.array, kv, offset: int, tile_width: int,
            on_progress=None) -> mx.array:
        """K3 layer-major prefill with exact attention/MLP NF12 lifetimes.

        The released operation order is unchanged. Only storage/materialization
        is split: attention tensors are decoded, consumed, and released before
        the router/shared-expert tensors become the live page. This keeps the
        authoritative q16 routed-expert pipeline from competing with a decoded
        whole-layer BF16 buffer.
        """
        from .kimi_linear import (
            DiskBackedAttnResSnapshots,
            _kda_attention,
            _kimi_dense_mlp_tiled,
            _kimi_moe_output,
            _mla_attention,
            apply_output_attn_res,
            attn_res_attention_input,
            attn_res_mlp_input,
        )

        n = self.cfg.num_hidden_layers
        total = int(x.shape[1])
        (
            self._layer_transient,
            self._layer_transient_margin,
        ) = _layer_transient_for_positions(
            total,
            getattr(
                self, "_prefill_layer_transient_by_positions", {}
            ).get(total, 0),
            getattr(self, "_decode_layer_transient", 0),
        )
        profiler = self._request_profiler
        if profiler is not None:
            profiler.begin_sweep(
                total, path="layer_stationary_kimi_k3_nf12_split"
            )
        if self.rc.kimi_k3_attnres_spill_dir:
            block_residual = DiskBackedAttnResSnapshots(
                self.rc.kimi_k3_attnres_spill_dir,
                write_tile_rows=tile_width,
            )
        else:
            block_residual = (
                []
                if self.rc.kimi_k3_fused_attnres_tile_size
                else mx.zeros(
                    (x.shape[0] * total, 0, x.shape[2]), dtype=x.dtype
                )
            )

        for i in range(n):
            self._select_layer_transient(total, i)

            cache_before = (
                profiler.cache_snapshot(self.cache)
                if profiler is not None
                else None
            )
            prefix = f"model.layers.{i}"
            layer_key = self._layer_key(i)
            attention_names, mlp_names = (
                self._k3_nf12_split_layer_names(i)
            )
            attention_key = f"{layer_key}.attn"
            mlp_key = f"{layer_key}.mlp"

            wait_started = time.perf_counter()
            if not self.cache.contains(attention_key):
                incoming = self._layer_fetch_bytes_estimate(
                    i, attention_names
                )
                self.cache.prepare_for(incoming)
                if self.governor is not None:
                    self.governor.reserve(incoming)
            attention_w = self.cache.get(
                attention_key, attention_names
            )
            attention_wait_s = time.perf_counter() - wait_started
            self.timer.add("weights_wait", attention_wait_s)

            # The MLP stream is authoritative for this same layer. Let its
            # mapped decode overlap attention only when the decoded page can
            # survive under the current cache budget.
            if self.prefetcher:
                self.prefetcher.schedule(
                    mlp_key,
                    mlp_names,
                    page_size_hint=self._layer_fetch_bytes_estimate(
                        i, mlp_names
                    ),
                )

            active_before = mx.get_active_memory()
            mx.reset_peak_memory()
            compute_started = time.perf_counter()

            prefix_sum, block_residual, attention_input = (
                attn_res_attention_input(
                    x,
                    block_residual,
                    attention_w,
                    prefix,
                    self.cfg,
                    i,
                    fused_tile_size=(
                        self.rc.kimi_k3_fused_attnres_tile_size),
                )
            )
            tiles = []
            pos = 0
            while pos < total:
                end = min(pos + tile_width, total)
                if self.governor is not None and self._layer_transient:
                    self.governor.reserve(
                        self._layer_transient,
                        margin=self._layer_transient_margin,
                    )
                ht = attention_input[:, pos:end, :]
                attention_t0 = time.perf_counter()
                if i in self.cfg.full_attn_layers:
                    yt = _mla_attention(
                        ht,
                        attention_w,
                        prefix,
                        self.cfg,
                        kv,
                        i,
                        offset + pos,
                    )
                elif i in self.cfg.kda_layers:
                    kda_cache = getattr(kv, "kda_cache", None)
                    yt = _kda_attention(
                        ht,
                        attention_w,
                        prefix,
                        self.cfg,
                        kda_cache,
                        i,
                        native_fused_decode=(
                            self.rc.native_fused_deltanet_decode
                        ),
                        native_fused_prefill=(
                            self.rc.kimi_k3_native_fused_kda_prefill
                        ),
                        compiled_prefill=(
                            self.rc.kimi_k3_compiled_kda_prefill
                        ),
                        profile=profiler,
                    )
                else:
                    raise ValueError(
                        f"layer {i} is in neither cfg.full_attn_layers "
                        "nor cfg.kda_layers"
                    )
                mx.eval(yt)
                if profiler is not None and profiler.sync_substeps:
                    profiler.record_substep(
                        "attention",
                        i,
                        time.perf_counter() - attention_t0,
                        positions=end - pos,
                    )
                tiles.append(yt)
                pos = end
            attention_out = (
                tiles[0]
                if len(tiles) == 1
                else mx.concatenate(tiles, axis=1)
            )
            prefix_sum = (
                prefix_sum + attention_out
                if prefix_sum is not None
                else attention_out
            )
            mx.eval(prefix_sum)
            del attention_input, attention_out, tiles, attention_w
            self.cache.discard(attention_key, attention_names)

            mlp_wait_started = time.perf_counter()
            if not self.cache.contains(mlp_key):
                incoming = self._layer_fetch_bytes_estimate(i, mlp_names)
                self.cache.prepare_for(incoming)
                if self.governor is not None:
                    self.governor.reserve(incoming)
            mlp_w = self.cache.get(mlp_key, mlp_names)
            mlp_wait_s = time.perf_counter() - mlp_wait_started
            self.timer.add("weights_wait", mlp_wait_s)

            # The current MLP page is now authoritative and cannot be displaced
            # by speculation. Only at this boundary may the worker start the
            # next attention page, overlapping its sequential decode with the
            # much longer routed-expert MLP rather than delaying this layer's
            # MLP behind speculative work.
            if self.prefetcher and i + 1 < n:
                next_attention, _ = (
                    self._k3_nf12_split_layer_names(i + 1)
                )
                self.prefetcher.schedule(
                    f"{self._layer_key(i + 1)}.attn",
                    next_attention,
                    page_size_hint=self._layer_fetch_bytes_estimate(
                        i + 1, next_attention
                    ),
                )

            mlp_input = attn_res_mlp_input(
                prefix_sum,
                block_residual,
                mlp_w,
                prefix,
                self.cfg,
                fused_tile_size=(
                    self.rc.kimi_k3_fused_attnres_tile_size),
            )
            mlp_t0 = time.perf_counter()
            if i < self.cfg.first_k_dense_replace:
                mlp_out = _kimi_dense_mlp_tiled(
                    mlp_input,
                    mlp_w,
                    f"{prefix}.mlp",
                    self.cfg,
                    self.rc.kimi_k3_dense_mlp_tile_size,
                )
            else:
                if self.rc.expert_batch_prefetch:
                    self._expert_shared_overlap_layers += 1
                mlp_out = _kimi_moe_output(
                    mlp_input,
                    mlp_w,
                    prefix,
                    self.cfg,
                    i,
                    self._get_experts,
                    iter_expert_batches=self._iter_expert_batches,
                    profile=profiler,
                    overlap_shared_expert=(
                        self.rc.expert_batch_prefetch
                    ),
                )
            mx.eval(mlp_out)
            if profiler is not None and profiler.sync_substeps:
                profiler.record_substep(
                    "mlp",
                    i,
                    time.perf_counter() - mlp_t0,
                    positions=total,
                )
            x = prefix_sum + mlp_out
            mx.eval(x)
            total_compute_wall = time.perf_counter() - compute_started
            compute_s = max(0.0, total_compute_wall - mlp_wait_s)
            weight_wait_s = attention_wait_s + mlp_wait_s
            self.timer.add("layer_compute", compute_s)

            if profiler is not None:
                profiler.record_layer(
                    i,
                    positions=total,
                    weight_wait_s=weight_wait_s,
                    compute_s=compute_s,
                    cache_before=cache_before,
                    cache_after=profiler.cache_snapshot(self.cache),
                    layer_type=self._profile_layer_type(i),
                )
            if on_progress is not None:
                on_progress({
                    "phase": "prefill_layer",
                    "completed_layers": i + 1,
                    "total_layers": n,
                    "total_tokens": total,
                    "cache_source": "cold",
                })

            end_active = mx.get_active_memory()
            peak_active = mx.get_peak_memory()
            measured_transient = _resident_adjusted_transient(
                active_before, end_active, peak_active
            )
            self._last_k3_transient_observation = {
                "layer": i,
                "signature": self._transient_layer_signature(i),
                "start_active_bytes": int(active_before),
                "end_active_bytes": int(end_active),
                "peak_active_bytes": int(peak_active),
                "measured_transient_bytes": int(measured_transient),
            }
            self._record_layer_transient(
                total, i, measured_transient
            )
            self._note_true_peak()
            del mlp_input, mlp_out, mlp_w, prefix_sum
            self.cache.discard(mlp_key, mlp_names)

        self._restore_aggregate_layer_transient(total)
        result = apply_output_attn_res(
            x,
            {
                "model.output_attn_res_proj.weight": (
                    self._output_attn_res_proj_w
                ),
                "model.output_attn_res_norm.weight": (
                    self._output_attn_res_norm_w
                ),
            },
            block_residual,
            self.cfg,
            fused_tile_size=(
                self.rc.kimi_k3_fused_attnres_tile_size),
        )
        if isinstance(block_residual, DiskBackedAttnResSnapshots):
            self._last_k3_attnres_spill_stats = block_residual.stats()
            block_residual.close()
        return result

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

    def consume_serial_kda_endpoint(
        self, fed_positions: int | None
    ):
        """Consume one retained recurrent endpoint from the last verify call.

        ``fed_positions`` is one-based within that verifier window. Passing
        ``None`` discards every retained endpoint. The final/full-window KDA
        state already lives in ``kv.kda_cache`` and is deliberately not
        duplicated here, so only shorter accepted prefixes are selectable.
        """
        endpoints = getattr(self, "_serial_kda_endpoints", None)
        self._serial_kda_endpoints = None
        self._serial_kda_endpoint_retained_bytes = 0
        if fed_positions is None or endpoints is None:
            return None
        count = int(fed_positions)
        if not 1 <= count <= len(endpoints):
            raise ValueError(
                f"retained KDA endpoint {count} is outside "
                f"[1, {len(endpoints)}]"
            )
        selected = endpoints[count - 1]
        return selected if selected.nbytes() > 0 else None

    def consume_serial_kda_factors(self):
        """Consume compact factors retained by the last serial verifier."""
        factors = getattr(self, "_serial_kda_factors", None)
        self._serial_kda_factors = None
        self._serial_kda_factor_retained_bytes = 0
        return factors

    def consume_serial_qwen4_endpoint(
        self, fed_positions: int | None,
    ):
        """Consume one exact QSA/PLE endpoint from the last verify window.

        QSA keys are append-only and are trimmed from the installed full-window
        state, but PLE convolution/context state cannot be
        reconstructed by trimming a rejected suffix.  Qwen4 speculative
        verification therefore retains every strict prefix alongside the
        existing KDA endpoint window.  The full-window endpoint remains
        installed in ``kv.qwen4_cache`` and is not duplicated here.
        """
        endpoints = getattr(self, "_serial_qwen4_endpoints", None)
        self._serial_qwen4_endpoints = None
        self._serial_qwen4_endpoint_retained_bytes = 0
        if fed_positions is None or endpoints is None:
            return None
        count = int(fed_positions)
        if not 1 <= count <= len(endpoints):
            raise ValueError(
                f"retained Qwen4 endpoint {count} is outside "
                f"[1, {len(endpoints)}]")
        return endpoints[count - 1]

    def forward_tokens_serial_positions(
        self, tokens: list[int], kv, tap_layers=None, *,
        capture_kda_endpoints: bool = False,
        capture_kda_factors: bool = False,
        capture_qwen4_endpoints: bool = False,
    ) -> mx.array:
        """Exact dense verification with one weight sweep for many positions.

        Batched ``(L, hidden)`` GEMMs can choose different reduction kernels
        from ordinary one-token greedy decode and were observed to move Qwen-7B
        tokens during speculative verification. Process positions serially at
        every layer instead, but keep the loop layer-major so a streamed target
        fetches each layer only once for the complete verify window.
        """
        # Never leave a prior verifier's large recurrent snapshots resident.
        # A successful caller must consume the desired endpoint immediately.
        self.consume_serial_kda_endpoint(None)
        self.consume_serial_kda_factors()
        self.consume_serial_qwen4_endpoint(None)
        if capture_kda_endpoints and capture_kda_factors:
            raise ValueError(
                "capture either dense KDA endpoints or compact factors, not both")

        glm_family = self.cfg.model_type in ("glm_moe_dsa", "kimi_k25", "glm4_moe_lite")
        glm5_family = self.cfg.model_type == "glm5_next"
        qwen_family = self.cfg.model_type in ("qwen3_5", "qwen3_5_moe")
        qwen4_family = self.cfg.model_type == "qwen4_exp"
        kimi_family = self.cfg.model_type == "kimi_linear"
        # F128: kimi_k3 gets its OWN branch below (kimi_k3_family), not
        # folded into kimi_family -- block_residual's per-layer boundary
        # snapshot must span ALL positions in this verify window at once,
        # not one position at a time like attention/MLP below do, so it
        # needs different bookkeeping around the same per-position calls,
        # not just a reused _kimi_linear_attention_residual/_mlp_residual
        # dispatch.
        kimi_k3_family = self.cfg.model_type == "kimi_k3"
        lfm2_family = self.cfg.model_type == "lfm2"
        if not glm_family and not glm5_family and not qwen_family and not qwen4_family and not kimi_family and not kimi_k3_family and (
                self.cfg.num_experts or self.cfg.model_type == "gpt_oss"):
            # F94: layer_runner.run_block (this function's per-layer call
            # below) is a plain dense-transformer block with no awareness of
            # the hybrid DeltaNet/full-attention layer_types these model
            # types use -- it silently looked up "model.layers.N.self_attn.*"
            # tensor names that don't exist on a linear_attention layer,
            # KeyError'ing rather than misrouting quietly. forward_tokens
            # (via _sweep, which DOES have correct model_type dispatch) is
            # the working alternative for these targets -- see
            # runtime/qwen35_mtp.py.
            #
            # F113 (2026-07-25): GLM-family (glm_moe_dsa/kimi_k25/
            # glm4_moe_lite) targets get the real fix instead of exclusion
            # -- see the glm_family per-position dispatch branch below,
            # which reuses _glm_attention_residual/_glm_mlp_residual
            # (already split out for F35) so MoE routing/MLA attention runs
            # at the SAME one-position-at-a-time granularity ordinary
            # decode uses, matching this function's whole purpose (exact
            # per-position match with true sequential decode) instead of
            # the batched-GEMM-can-diverge risk forward_tokens carries.
            #
            # F113 follow-on (2026-07-25, later): qwen3_5/qwen3_5_moe
            # (Qwen3.5-4B/9B, Qwen3.6-27B/35B-A3B) get the same real fix,
            # reusing _qwen35_attention_residual/_qwen35_mlp_residual
            # (split out for F106). This ALSO makes DeltaNet's recurrent
            # state (kda_cache) update position-by-position through this
            # function exactly as real sequential decode would -- the
            # per-position dispatch isn't just avoiding batched-GEMM
            # divergence here, it's the SAME mechanism that makes ordinary
            # decode's own KDA state evolution correct, applied to a
            # verify window instead of one live token at a time.
            #
            # F113 follow-on (2026-07-26, Kimi K3 readiness): kimi_linear
            # (3:1 KDA-to-MLA hybrid, same recurrent-state shape as
            # qwen_family plus GLM-shaped MLA/MoE on its full_attn_layers)
            # reuses _kimi_linear_attention_residual/_kimi_linear_mlp_residual
            # (already split out for F35-prep, kimi_linear.py) unchanged --
            # that split already internally dispatches KDA vs MLA per
            # layer via cfg.full_attn_layers/cfg.kda_layers, so this
            # function only needed to call it per-position, mirroring the
            # qwen_family branch above.
            raise ValueError(
                "serial-position verification currently supports dense models only")
        if not tokens:
            raise ValueError("serial-position verification needs at least one token")
        if len(tokens) == 1:
            return self.forward_tokens(tokens, kv, tap_layers=tap_layers)
        if qwen_family:
            self._suspend_qwen35_serial_verify_lm_head()
        elif qwen4_family:
            # The draft path synchronizes/detaches q before entering this
            # target-authoritative sweep.  Drop only the output-head lease;
            # _final_logits restores the same BF16 page after all target trunk
            # layers have completed.
            self._suspend_qwen4_serial_verify_lm_head()

        offset = kv.offset
        verifier_positions = len(tokens)
        profiler = self._request_profiler
        if profiler is not None:
            profiler.begin_sweep(
                verifier_positions, path="serial_positions")
        self._tap_hidden = {}
        # Every block invocation below has the ordinary one-position decode
        # shape even though several positions are retained across the
        # layer-major sweep.  Do not inherit a batched-prefill scratch maximum.
        self._layer_transient, self._layer_transient_margin = (
            _layer_transient_for_positions(
                1,
                getattr(self, "_prefill_layer_transient", 0),
                getattr(self, "_decode_layer_transient", 0)))
        tapset = set(tap_layers) if tap_layers is not None else None
        embedded = self._embed(list(tokens))
        positions = [embedded[:, i:i + 1, :] for i in range(len(tokens))]
        n = self.cfg.num_hidden_layers
        serial_kda_endpoints = None
        serial_qwen4_endpoints = None
        factor_source = None
        if capture_kda_endpoints:
            source_kda = getattr(kv, "kda_cache", None)
            if source_kda is None:
                raise ValueError(
                    "KDA endpoint capture requires kv.kda_cache"
                )
            from .kda_state import KDAStateCache

            # The complete-window endpoint remains installed in source_kda.
            # Retain only the strict prefixes that a rejection/stop can select.
            serial_kda_endpoints = [
                KDAStateCache(n) for _ in range(len(tokens) - 1)
            ]
        elif capture_kda_factors:
            factor_source = getattr(kv, "kda_cache", None)
            if factor_source is None:
                raise ValueError(
                    "KDA factor capture requires kv.kda_cache")
            factor_source.begin_factor_capture()
        if glm5_family:
            if capture_kda_endpoints:
                raise ValueError(
                    "GLM-5.3 verifier uses compact KDA factors, not dense "
                    "per-position endpoints")
            x_all = self._layer_stationary_glm5_next_sweep(
                embedded, kv, offset, tile_width=1,
                tap_layers=tap_layers,
                incremental_expanded_mla=False)
            positions = [
                x_all[:, position:position + 1, :]
                for position in range(x_all.shape[1])
            ]
            head = self._lm_head_weight()
            from .lm_head_stream import StreamedLMHead

            if isinstance(head, StreamedLMHead):
                normalized = mx.concatenate([
                    mx.fast.rms_norm(
                        hidden, self._norm_w, self.cfg.rms_norm_eps)
                    for hidden in positions
                ], axis=1)
                result = head.logits_serial_rows(normalized)[0]
            else:
                logits = []
                for hidden in positions:
                    value = self._final_logits(hidden, head=head)
                    mx.eval(value)
                    logits.append(value)
                result = mx.stack(logits)
            mx.eval(result)
            self._h_window = x_all
            self._h_last = positions[-1]
            if factor_source is not None:
                self._serial_kda_factors = factor_source.finish_factor_capture(
                    len(tokens))
                self._serial_kda_factor_retained_bytes = (
                    self._serial_kda_factors.nbytes()
                    if self._serial_kda_factors is not None else 0)
            return result
        if capture_qwen4_endpoints:
            source_qwen4 = getattr(kv, "qwen4_cache", None)
            if source_qwen4 is None:
                raise ValueError(
                    "Qwen4 endpoint capture requires kv.qwen4_cache")
            from .qwen4_exp_state import Qwen4ExpStateCache

            serial_qwen4_endpoints = [
                Qwen4ExpStateCache(n) for _ in range(len(tokens) - 1)
            ]
        if glm_family:
            from .glm import _glm_attention_residual, _glm_mlp_residual
        if qwen_family:
            from .qwen35 import _qwen35_attention_residual, _qwen35_mlp_residual
        if qwen4_family:
            from .qwen4_exp import (
                qwen4_attention_residual,
                qwen4_mlp_from_group_batches,
                qwen4_mlp_from_groups,
                qwen4_mlp_route,
                qwen4_mlp_route_window_exact,
            )
        if kimi_family:
            from .kimi_linear import (
                _kimi_linear_attention_residual, _kimi_linear_mlp_residual)
        if lfm2_family:
            from .lfm2 import _lfm2_mlp_residual, _lfm2_operator_residual
        if kimi_k3_family:
            from .kimi_linear import (
                _apply_attn_res, _kda_attention, _mla_attention,
                _kimi_dense_mlp, _kimi_moe_output)
            block_residual = (
                []
                if self.rc.kimi_k3_fused_attnres_tile_size
                else mx.zeros(
                    (
                        embedded.shape[0] * len(tokens),
                        0,
                        embedded.shape[2],
                    ),
                    dtype=embedded.dtype,
                )
            )

        def serial_expert_batches(position: int):
            """Translate one-position-local route rows into verify-window rows.

            Each architecture helper below sees a ``(1, 1, hidden)`` slice,
            so its router labels the only row as position zero. Provisional
            speculative bookkeeping, however, must know which window position
            produced the route so rejected-tail observations are not committed.
            """
            def batches(layer, expert_ids, positions=None):
                shifted = offset_expert_route_positions(
                    positions, position
                )
                return self._iter_expert_batches(
                    layer, expert_ids, positions=shifted
                )

            return batches

        def capture_kda_position(layer: int, position: int) -> None:
            """Retain this layer's exact recurrent endpoint for one prefix."""
            if (
                serial_kda_endpoints is None
                or position >= len(serial_kda_endpoints)
            ):
                return
            source = getattr(kv, "kda_cache", None)
            if source is None:
                return
            state = source.state(layer)
            history = source.conv_history(layer)
            if state is None and history is None:
                return
            endpoint = serial_kda_endpoints[position]
            if state is not None:
                endpoint.set_state(layer, state)
            if history is not None:
                endpoint.set_conv_history(layer, tuple(history))

        def capture_qwen4_position(layer: int, position: int) -> None:
            """Retain this layer's exact QSA/PLE state for one strict prefix."""
            if (
                serial_qwen4_endpoints is None
                or position >= len(serial_qwen4_endpoints)
            ):
                return
            source = getattr(kv, "qwen4_cache", None)
            if source is None:
                return
            endpoint = serial_qwen4_endpoints[position]
            if source.ple_conv[layer] is not None:
                endpoint.ple_conv[layer] = source.ple_conv[layer]
                endpoint.ple_context[layer] = source.ple_context[layer]
                endpoint.ple_lengths[layer] = source.ple_lengths[layer]

        for layer in range(n):
            cache_before = (
                profiler.cache_snapshot(self.cache)
                if profiler is not None else None)
            self._select_serial_verify_layer_transient(
                verifier_positions, layer
            )
            dspark_prefetch_plan = getattr(
                self, "_dspark_expert_prefetch_plan", None)
            if dspark_prefetch_plan is not None:
                dspark_prefetch_plan.schedule_before_layer(
                    self,
                    layer,
                    int(getattr(
                        self, "_dspark_expert_prefetch_depth", 0)),
                )
            page_prepare_t0 = time.perf_counter()
            self._prepare_serial_verify_layer_page(layer)
            if qwen_family:
                self._qwen35_serial_verify_page_prepare_s += (
                    time.perf_counter() - page_prepare_t0)
            elif qwen4_family:
                self._qwen4_serial_verify_page_prepare_s += (
                    time.perf_counter() - page_prepare_t0)
            t0 = time.perf_counter()
            weights = self.cache.get(
                self._layer_key(layer), self._layer_names(layer))
            weight_wait_s = time.perf_counter() - t0
            self.timer.add("weights_wait", weight_wait_s)
            if qwen_family:
                self._qwen35_serial_verify_weight_wait_s += weight_wait_s
            elif qwen4_family:
                self._qwen4_serial_verify_weight_wait_s += weight_wait_s
            if self.governor is not None and self._layer_transient:
                reserve_t0 = time.perf_counter()
                self.governor.reserve(
                    self._layer_transient,
                    margin=self._layer_transient_margin,
                    reason="serial-verify-transient")
                if qwen_family:
                    self._qwen35_serial_verify_reserve_s += (
                        time.perf_counter() - reserve_t0)
                elif qwen4_family:
                    self._qwen4_serial_verify_reserve_s += (
                        time.perf_counter() - reserve_t0)
            # Schedule future I/O only after the current demand page and its
            # compute transient have passed live admission.  Scheduling two
            # ~203 MB Huihui pages first made the current reservation inherit
            # their speculative allocations under tight headroom, repeatedly
            # collapsing the cache and pausing the prefetcher.  This ordering
            # still overlaps next-layer reads with all current-layer compute;
            # it merely prevents future work from invalidating the admission
            # proof for the work that must run now.
            if self.prefetcher:
                for nxt in range(
                        layer + 1,
                        min(layer + 1 + self.rc.prefetch_depth, n)):
                    self.prefetcher.schedule(
                        self._layer_key(nxt), self._layer_names(nxt))

            layer_compute_t0 = time.perf_counter()
            active_before = mx.get_active_memory()
            mx.reset_peak_memory()
            if kimi_k3_family:
                # F128: block_residual's per-layer boundary snapshot must
                # span ALL positions in this verify window at once (it is
                # built ONCE per layer, not once per position -- see
                # attn_res_wrap_layer's own docstring), so the AttnRes
                # pre/post mixing (_apply_attn_res) is batched across all N
                # positions here. This does not weaken this function's
                # "avoid batched-GEMM divergence" guarantee: _apply_attn_res
                # has no hidden_size-sized reduction (softmax/matmul over
                # the tiny num_blocks axis only, per-position independent),
                # unlike a real hidden_size GEMM. Attention and MLP/MoE
                # still run one position at a time, in order, exactly like
                # every other family above.
                prefix = f"model.layers.{layer}"
                x_layer = mx.concatenate(positions, axis=1)
                B, N, H = x_layer.shape
                prefix_sum_layer = x_layer
                hidden_layer = x_layer
                block_count = (
                    len(block_residual)
                    if isinstance(block_residual, list)
                    else block_residual.shape[1]
                )
                if block_count > 0:
                    hidden_layer = _apply_attn_res(
                        prefix_sum_layer.reshape(-1, H), block_residual,
                        weights[f"{prefix}.self_attention_res_proj.weight"],
                        weights[f"{prefix}.self_attention_res_norm.weight"],
                        self.cfg.rms_norm_eps,
                        fused_tile_size=(
                            self.rc.kimi_k3_fused_attnres_tile_size),
                    ).reshape(B, N, H)
                if layer % self.cfg.attn_res_block_size == 0:
                    snapshot = prefix_sum_layer.reshape(-1, H)
                    if isinstance(block_residual, list):
                        block_residual.append(snapshot)
                    else:
                        block_residual = mx.concatenate(
                            [block_residual, snapshot[:, None, :]],
                            axis=1,
                        )
                    prefix_sum_layer = None

                attn_outputs = []
                for position in range(N):
                    h_normed = mx.fast.rms_norm(
                        hidden_layer[:, position:position + 1, :],
                        weights[f"{prefix}.input_layernorm.weight"],
                        self.cfg.rms_norm_eps)
                    if layer in self.cfg.full_attn_layers:
                        attn_out = _mla_attention(
                            h_normed, weights, prefix, self.cfg, kv, layer,
                            offset + position)
                    else:
                        kda_cache = getattr(kv, "kda_cache", None)
                        attn_out = _kda_attention(
                            h_normed, weights, prefix, self.cfg, kda_cache, layer)
                        capture_kda_position(layer, position)
                    attn_outputs.append(attn_out)
                attn_out_layer = mx.concatenate(attn_outputs, axis=1)
                prefix_sum_layer = (
                    (prefix_sum_layer + attn_out_layer)
                    if prefix_sum_layer is not None else attn_out_layer)

                hidden_layer2 = _apply_attn_res(
                    prefix_sum_layer.reshape(-1, H), block_residual,
                    weights[f"{prefix}.mlp_res_proj.weight"],
                    weights[f"{prefix}.mlp_res_norm.weight"],
                    self.cfg.rms_norm_eps,
                    fused_tile_size=(
                        self.rc.kimi_k3_fused_attnres_tile_size),
                ).reshape(B, N, H)

                mlp_outputs = []
                for position in range(N):
                    h2_normed = mx.fast.rms_norm(
                        hidden_layer2[:, position:position + 1, :],
                        weights[f"{prefix}.post_attention_layernorm.weight"],
                        self.cfg.rms_norm_eps)
                    if layer < self.cfg.first_k_dense_replace:
                        mlp_out = _kimi_dense_mlp(
                            h2_normed, weights, f"{prefix}.mlp", self.cfg)
                    else:
                        mlp_out = _kimi_moe_output(
                            h2_normed, weights, prefix, self.cfg, layer,
                            self._get_experts,
                            iter_expert_batches=serial_expert_batches(
                                position
                            ))
                    mlp_outputs.append(mlp_out)
                mlp_out_layer = mx.concatenate(mlp_outputs, axis=1)
                prefix_sum_layer = prefix_sum_layer + mlp_out_layer
                next_positions = [
                    prefix_sum_layer[:, p:p + 1, :] for p in range(N)]
            elif (
                qwen_family
                and self.rc.qwen35_serial_verify_batched_mlp
                and not self.cfg.num_experts
            ):
                # Attention remains strictly sequential because DeltaNet state
                # and full-attention KV advance position by position.  The
                # dense MLP has no cross-position dependency: each row sees
                # only its own post-attention residual and the same weights.
                # Concatenate only after every canonical attention call, then
                # split immediately after one SwiGLU projection group.
                prefix = f"model.layers.{layer}"
                attn_positions = []
                for position, hidden in enumerate(positions):
                    attn_out = _qwen35_attention_residual(
                        hidden, weights, prefix, self.cfg, kv, layer,
                        offset + position,
                        chunked_delta_prefill=(
                            self.rc.qwen_chunked_delta_prefill),
                        compiled_delta_prefill=(
                            self.rc.qwen_compiled_delta_prefill),
                        native_fused_delta_prefill=(
                            self.rc.qwen_native_fused_delta_prefill),
                        zmlx_fused_decode=(
                            self.rc.zmlx_fused_deltanet_decode),
                        native_fused_decode=(
                            self.rc.native_fused_deltanet_decode),
                    )
                    capture_kda_position(layer, position)
                    attn_positions.append(attn_out)
                batched_mlp_t0 = time.perf_counter()
                batched_hidden = _qwen35_mlp_residual(
                    mx.concatenate(attn_positions, axis=1),
                    weights,
                    prefix,
                    self.cfg,
                    layer,
                    self._get_experts,
                )
                mx.eval(batched_hidden)
                self._qwen35_serial_verify_batched_mlp_s += (
                    time.perf_counter() - batched_mlp_t0)
                self._qwen35_serial_verify_batched_mlp_layers += 1
                self._qwen35_serial_verify_batched_mlp_positions += len(
                    positions)
                next_positions = [
                    batched_hidden[:, position:position + 1, :]
                    for position in range(len(positions))
                ]
            elif qwen4_family:
                # The recurrent PLE/QSA/DeltaNet half remains canonical and
                # position-serial.  Once all rows have been routed, fetch the
                # immutable union of their exact BF16 expert pages once for
                # this layer.  Each row still executes its experts separately
                # in ascending order with the same materialization boundary as
                # ordinary one-token decode; this only removes duplicate I/O.
                prefix = f"model.layers.{layer}"
                attn_positions = []
                for position, hidden in enumerate(positions):
                    attn_out = qwen4_attention_residual(
                        hidden,
                        (tokens[position],),
                        weights,
                        prefix,
                        self.cfg,
                        kv,
                        layer,
                        offset + position,
                        row_store=self._qwen4_ple_rows,
                        compiled_delta_prefill=(
                            self.rc.qwen_compiled_delta_prefill),
                        native_fused_delta_prefill=(
                            self.rc.qwen_native_fused_delta_prefill),
                    )
                    capture_kda_position(layer, position)
                    capture_qwen4_position(layer, position)
                    attn_positions.append(attn_out)

                exact_bf16_stats: dict[str, int] = {}
                if self.rc.qwen4_serial_verify_exact_bf16_gemv:
                    routes = qwen4_mlp_route_window_exact(
                        attn_positions,
                        weights,
                        prefix,
                        self.cfg,
                        layer,
                        stats=exact_bf16_stats,
                    )
                else:
                    routes = [
                        qwen4_mlp_route(
                            hidden, weights, prefix, self.cfg, layer)
                        for hidden in attn_positions
                    ]
                positions_by_expert: dict[int, list[int]] = {}
                expert_slots = 0
                for position, route in enumerate(routes):
                    groups = route[3]
                    expert_slots += len(groups)
                    for expert in groups:
                        positions_by_expert.setdefault(expert, []).append(
                            position)
                expert_ids = sorted(positions_by_expert)
                expert_batch_prefetch_active = bool(
                    self._expert_batch_executor is not None
                    and self._expert_batch_prefetch_active)
                if expert_batch_prefetch_active:
                    expert_wait_before = float(
                        self.timer.totals.get("expert_wait", 0.0))
                    next_positions = qwen4_mlp_from_group_batches(
                        routes,
                        self._iter_expert_batches(
                            layer, expert_ids,
                            positions=positions_by_expert,
                        ),
                        weights,
                        prefix,
                        exact_bf16=(
                            self.rc.qwen4_serial_verify_exact_bf16_gemv),
                        exact_stats=exact_bf16_stats,
                    )
                    self._qwen4_serial_verify_union_fetch_s += max(
                        0.0,
                        float(self.timer.totals.get("expert_wait", 0.0))
                        - expert_wait_before,
                    )
                    self._qwen4_serial_verify_pipelined_expert_layers += 1
                else:
                    union_fetch_t0 = time.perf_counter()
                    experts = self._get_experts(
                        layer, expert_ids, positions=positions_by_expert)
                    self._qwen4_serial_verify_union_fetch_s += (
                        time.perf_counter() - union_fetch_t0)
                    if self.rc.qwen4_serial_verify_exact_bf16_gemv:
                        next_positions = qwen4_mlp_from_group_batches(
                            routes,
                            iter(((expert_ids, experts),)),
                            weights,
                            prefix,
                            exact_bf16=True,
                            exact_stats=exact_bf16_stats,
                        )
                    else:
                        next_positions = [
                            qwen4_mlp_from_groups(
                                route, experts, weights, prefix)
                            for route in routes
                        ]
                self._qwen4_serial_verify_union_layers += 1
                self._qwen4_serial_verify_expert_slots += expert_slots
                self._qwen4_serial_verify_union_experts += len(expert_ids)
                self._qwen4_serial_verify_expert_pages_avoided += max(
                    0, expert_slots - len(expert_ids))
                self._qwen4_serial_verify_exact_bf16_calls += int(
                    exact_bf16_stats.get("calls", 0))
                self._qwen4_serial_verify_exact_bf16_rows += int(
                    exact_bf16_stats.get("rows", 0))
                self._qwen4_serial_verify_exact_bf16_fallback_calls += int(
                    exact_bf16_stats.get("fallback_calls", 0))
                for reason in (
                    "unavailable", "rank", "dtype", "inner_dimension",
                    "empty_batch", "singleton_window", "window_too_wide",
                    "output_geometry", "skinny_output", "unknown",
                ):
                    value = int(exact_bf16_stats.get(
                        f"fallback_{reason}_calls", 0))
                    if value:
                        reasons = (
                            self._qwen4_serial_verify_exact_bf16_fallback_reasons)
                        reasons[reason] = reasons.get(reason, 0) + value
                mx.eval(*next_positions)
                if not expert_batch_prefetch_active:
                    del experts
                del routes, attn_positions
            else:
                next_positions = []
                for position, hidden in enumerate(positions):
                    if glm_family:
                        # F113: MLA attention + MoE routing/experts computed at
                        # the SAME one-position-at-a-time granularity ordinary
                        # decode uses -- routing for position i sees only
                        # position i's own hidden state, exactly matching what
                        # true sequential decode would compute, unlike a
                        # batched multi-position routing call.
                        prefix = f"model.layers.{layer}"
                        attn_out = _glm_attention_residual(
                            hidden, weights, prefix, self.cfg, kv, layer,
                            offset + position)
                        hidden = _glm_mlp_residual(
                            attn_out, weights, prefix, self.cfg, layer,
                            self._get_experts,
                            iter_expert_batches=serial_expert_batches(
                                position
                            ))
                    elif qwen_family:
                        # F113 follow-on: DeltaNet-or-full-attention + MoE (if
                        # any) computed one position at a time, in order --
                        # kv.kda_cache (when present) updates exactly as it
                        # would during real sequential decode, since this IS
                        # a real per-position sequential call, not a batched
                        # multi-position one.
                        prefix = f"model.layers.{layer}"
                        attn_out = _qwen35_attention_residual(
                            hidden, weights, prefix, self.cfg, kv, layer,
                            offset + position,
                            chunked_delta_prefill=(
                                self.rc.qwen_chunked_delta_prefill),
                            compiled_delta_prefill=(
                                self.rc.qwen_compiled_delta_prefill),
                            native_fused_delta_prefill=(
                                self.rc.qwen_native_fused_delta_prefill),
                            zmlx_fused_decode=(
                                self.rc.zmlx_fused_deltanet_decode),
                            native_fused_decode=(
                                self.rc.native_fused_deltanet_decode))
                        capture_kda_position(layer, position)
                        hidden = _qwen35_mlp_residual(
                            attn_out, weights, prefix, self.cfg, layer,
                            self._get_experts,
                            iter_expert_batches=serial_expert_batches(
                                position
                            ))
                    elif lfm2_family:
                        # F203: same one-position-at-a-time contract as
                        # qwen_family. The short-conv layers advance their
                        # conv_L_cache-1 history exactly as real sequential
                        # decode would, because this IS a sequential call; the
                        # endpoint snapshot between the two halves is what a
                        # partial rejection restores.
                        prefix = f"model.layers.{layer}"
                        attn_out = _lfm2_operator_residual(
                            hidden, weights, prefix, self.cfg, kv, layer,
                            offset + position,
                            getattr(kv, "kda_cache", None))
                        capture_kda_position(layer, position)
                        hidden = _lfm2_mlp_residual(
                            attn_out, weights, prefix, self.cfg)
                    elif kimi_family:
                        # F113 follow-on (Kimi K3 readiness): same one-
                        # position-at-a-time contract as qwen_family above --
                        # _kimi_linear_attention_residual itself picks KDA vs
                        # MLA per layer (cfg.kda_layers/full_attn_layers), so
                        # kda_cache and MLA KV both evolve exactly as real
                        # sequential decode would.
                        prefix = f"model.layers.{layer}"
                        attn_out = _kimi_linear_attention_residual(
                            hidden, weights, prefix, self.cfg, kv, layer,
                            offset + position)
                        capture_kda_position(layer, position)
                        hidden = _kimi_linear_mlp_residual(
                            attn_out, weights, prefix, self.cfg, layer,
                            self._get_experts,
                            iter_expert_batches=serial_expert_batches(
                                position
                            ))
                    else:
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
            compute_s = time.perf_counter() - layer_compute_t0
            if qwen_family:
                if self.cfg.layer_types[layer] == "linear_attention":
                    self._qwen35_serial_verify_linear_layer_compute_s += (
                        compute_s)
                else:
                    self._qwen35_serial_verify_full_layer_compute_s += (
                        compute_s)
            elif qwen4_family:
                if self.cfg.layer_types[layer] == "linear_attention":
                    self._qwen4_serial_verify_linear_compute_s += compute_s
                    self._qwen4_serial_verify_linear_layers += 1
                else:
                    self._qwen4_serial_verify_full_compute_s += compute_s
                    self._qwen4_serial_verify_full_layers += 1
            if profiler is not None:
                profiler.record_layer(
                    layer,
                    positions=verifier_positions,
                    weight_wait_s=weight_wait_s,
                    compute_s=compute_s,
                    cache_before=cache_before,
                    cache_after=profiler.cache_snapshot(self.cache),
                    layer_type=self._profile_layer_type(layer),
                )
            positions = next_positions
            if tapset is not None and layer in tapset:
                # Preserve the same post-layer residual stream exposed by
                # _sweep(tap_layers=...).  Concatenating only after the
                # one-token-shaped layer calls have completed cannot change
                # verifier arithmetic, while DSpark receives one ordinary
                # (1, positions, hidden) context tensor per requested layer.
                self._tap_hidden[layer] = mx.concatenate(positions, axis=1)
            self._record_serial_verify_layer_transient(
                verifier_positions, layer,
                _resident_adjusted_transient(
                    active_before, mx.get_active_memory(),
                    mx.get_peak_memory()))
            self._note_true_peak()
            del weights

        self._layer_transient = max(
            (
                value
                for (width, _signature), value
                in self._serial_verify_layer_transient.items()
                if width == verifier_positions
            ),
            default=0,
        )
        self._layer_transient_margin = _layer_transient_reserve_margin(1)
        if kimi_k3_family:
            from .kimi_linear import apply_output_attn_res

            x_all = mx.concatenate(positions, axis=1)
            x_all = apply_output_attn_res(
                x_all, {
                    "model.output_attn_res_proj.weight": self._output_attn_res_proj_w,
                    "model.output_attn_res_norm.weight": self._output_attn_res_norm_w,
                }, block_residual, self.cfg,
                fused_tile_size=(
                    self.rc.kimi_k3_fused_attnres_tile_size))
            mx.eval(x_all)
            positions = [x_all[:, p:p + 1, :] for p in range(x_all.shape[1])]

        head_t0 = time.perf_counter()
        head = self._lm_head_weight()
        from .lm_head_stream import StreamedLMHead

        if isinstance(head, StreamedLMHead):
            # Preserve ordinary one-token contraction shapes while sharing
            # each physical LM-head block read across the complete verifier
            # window. A batched matmul is deliberately not substituted here:
            # this method exists because its reduction choice can move logits.
            if self.cfg.model_type == "qwen4_exp":
                from .qwen4_exp import final_hidden

                normalized = mx.concatenate([
                    final_hidden(
                        hidden, self._qwen4_final_mixer_w, self.cfg)
                    for hidden in positions
                ], axis=1)
            elif self.cfg.model_type in ("qwen3_5_moe", "qwen3_5"):
                from .qwen35 import qwen35_rms_norm

                normalize = qwen35_rms_norm
                normalized = mx.concatenate(
                    [
                        normalize(
                            hidden, self._norm_w, self.cfg.rms_norm_eps
                        )
                        for hidden in positions
                    ],
                    axis=1,
                )
            else:
                normalize = mx.fast.rms_norm
                normalized = mx.concatenate(
                    [
                        normalize(
                            hidden, self._norm_w, self.cfg.rms_norm_eps
                        )
                        for hidden in positions
                    ],
                    axis=1,
                )
            result = head.logits_serial_rows(normalized)[0]
        else:
            logits = []
            for hidden in positions:
                value = self._final_logits(hidden, head=head)
                mx.eval(value)
                logits.append(value)
            result = mx.stack(logits)
        mx.eval(result)
        if qwen_family:
            self._qwen35_serial_verify_head_s = float(getattr(
                self, "_qwen35_serial_verify_head_s", 0.0)) + (
                    time.perf_counter() - head_t0)
        elif qwen4_family:
            self._qwen4_serial_verify_head_s += (
                time.perf_counter() - head_t0)
        self._h_window = mx.concatenate(positions, axis=1)
        self._h_last = positions[-1]
        self._serial_kda_endpoints = serial_kda_endpoints
        self._serial_kda_endpoint_retained_bytes = (
            sum(endpoint.nbytes() for endpoint in serial_kda_endpoints)
            if serial_kda_endpoints is not None
            else 0
        )
        self._serial_qwen4_endpoints = serial_qwen4_endpoints
        self._serial_qwen4_endpoint_retained_bytes = (
            sum(endpoint.nbytes() for endpoint in serial_qwen4_endpoints)
            if serial_qwen4_endpoints is not None
            else 0
        )
        if factor_source is not None:
            self._serial_kda_factors = factor_source.finish_factor_capture(
                len(tokens))
            self._serial_kda_factor_retained_bytes = (
                self._serial_kda_factors.nbytes()
                if self._serial_kda_factors is not None else 0
            )
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
            scale_sidecar = getattr(self.store, "k3_scale_sidecar", None)
            nf12_sidecar = getattr(self.store, "bf16_nf12_sidecar", None)
            scale_sidecar_identity = (
                getattr(scale_sidecar, "generation_dir", Path("none")).name
                if scale_sidecar is not None else "none"
            )
            nf12_sidecar_identity = (
                getattr(nf12_sidecar, "generation_dir", Path("none")).name
                if nf12_sidecar is not None else "none"
            )
            arithmetic = (
                f"abs{int(self.rc.mla_absorbed_decode)}"
                f"glm53sparseabs{int(self.rc.glm53_sparse_absorbed_mla)}"
                f"glm53sparsefused{int(self.rc.glm53_sparse_fused_attention)}"
                f"glm53sparsekvint8{int(self.rc.glm53_sparse_fused_kv_int8)}"
                f"glm53coalexpert{int(
                    self.rc.glm53_coalesced_expert_positions)}"
                f"glm53coalexpertmax{
                    self.rc.glm53_coalesced_expert_max_positions}"
                f"glm53poolcache{int(self.rc.glm53_incremental_dsa_pool)}"
                f"glm53compiledkda{int(self.rc.glm53_compiled_kda_prefill)}"
                f"glm53compiledkdaseg{self.rc.glm53_compiled_kda_segment}"
                f"glm53nativekda{int(
                    self.rc.glm53_native_fused_kda_prefill)}"
                f"glm53hostspool{int(
                    self.rc.glm53_layer_stationary_host_spool)}"
                f"glmdsakeytile{self.rc.glm_dsa_key_tile_size}"
                f"glmdsaindexstep{self.rc.glm_dsa_index_step_size}"
                f"glmdsaindexprealloc{int(self.rc.glm_dsa_index_preallocate)}"
                f"glmdsaquerytile{self.rc.glm_dsa_selection_query_tile_size}"
                f"glmdsadensemlptile{self.rc.glm_dsa_dense_mlp_tile_size}"
                f"glmdsaabsorbed{int(self.rc.glm_dsa_sparse_absorbed_mla)}"
                f"glmdsaspill{int(bool(self.rc.glm_dsa_mla_kv_spill_dir))}"
                f"k3cmla{int(self.rc.kimi_k3_compressed_mla)}"
                f"k3abs{int(self.rc.kimi_k3_absorbed_mla)}"
                f"k3mlakt{self.rc.kimi_k3_mla_key_tile_size}"
                f"k3ar{self.rc.kimi_k3_fused_attnres_tile_size}"
                f"k3dmlp{self.rc.kimi_k3_dense_mlp_tile_size}"
                f"k3compiledkda{int(self.rc.kimi_k3_compiled_kda_prefill)}"
                f"k3nativekda{int(self.rc.kimi_k3_native_fused_kda_prefill)}"
                f"qwencompileddelta{int(self.rc.qwen_compiled_delta_prefill)}"
                f"qwennativeprefill{int(self.rc.qwen_native_fused_delta_prefill)}"
                f"qwenchunkeddelta{int(self.rc.qwen_chunked_delta_prefill)}"
                f"qwenserialbatchmlp{int(
                    self.rc.qwen35_serial_verify_batched_mlp)}"
                f"qwenpagedonline{int(
                    self.rc.qwen35_paged_online_attention)}"
                f"qwenpagedtile{self.rc.qwen35_paged_online_tile_positions}"
                f"qwenpagednative{int(
                    self.rc.qwen35_paged_online_page_native)}"
                f"k3scalesidecar{scale_sidecar_identity}"
                f"k3nf12sidecar{nf12_sidecar_identity}"
                f"k3nf12direct{int(self.rc.bf16_nf12_direct_linear)}"
                f"experttopk{tuple(self.cfg.expert_top_k_by_layer)}"
                f"expertprune{tuple(sorted(
                    (int(layer), tuple(experts))
                    for layer, experts in (
                        self.cfg.expert_prune_masks or {}).items()
                ))}"
                f"k3tilepolicy{self.rc.kimi_k3_prefill_tile_policy}"
                f"k3tilethreshold"
                f"{self.rc.kimi_k3_prefill_long_context_tokens}"
                f"k3shorttile{self.rc.kimi_k3_prefill_short_tile_width}"
                f"k3shortdmlp"
                f"{self.rc.kimi_k3_dense_mlp_short_tile_size}"
                f"k3longtile{self._k3_prefill_long_tile_width}"
                f"k3longdmlp{self._k3_dense_mlp_long_tile_size}"
                f"dead{int(self.rc.final_dead_token_elim)}"
                f"head{int(self.rc.stream_lm_head)}"
                f"tiedhead{int(self.rc.quantize_tied_lm_head)}"
                f"resident{int(self.rc.resident_fast_decode)}"
                f"residentprefill{self.rc.resident_fast_prefill_limit}"
                f"residentmoe{int(self.rc.resident_moe_decode)}"
                f"fswiglu{int(self.rc.fused_swiglu)}"
                f"chunk{self.rc.prefill_chunk_size}"
                f"qwenchunkceiling{self.rc.qwen35_prefill_chunk_ceiling}"
                f"layerstationary{int(self.rc.layer_stationary_prefill)}"
                f"qwensuffixdepth"
                f"{self.rc.qwen_lossy_suffix_prefill_early_layers}"
                f"qwenprefixtokens"
                f"{self.rc.qwen_lossy_suffix_prefill_prefix_tokens}"
                f"qwensuffixtokens"
                f"{self.rc.qwen_lossy_suffix_prefill_tokens}"
                f"lastsep{int(self.rc.prefill_last_token_separate)}"
                # The v2 schedule retains a layer-stationary prompt
                # endpoint's hidden state for logits instead of recomputing
                # its last token in a second streamed sweep. Batch shapes can
                # change floating reduction order, so old endpoint logits
                # must not share a persistence namespace.
                "layerstationaryendpointv2"
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
        k3_compressed_mla = bool(
            self.rc.kimi_k3_compressed_mla
            and self.cfg.model_type == "kimi_k3"
        )
        glm53_compressed_mla = bool(
            self.rc.mla_compressed_kv
            and self.cfg.model_type == "glm5_next"
        )
        glm_dsa_spilled_mla = bool(
            self.rc.mla_compressed_kv
            and self.cfg.model_type == "glm_moe_dsa"
            and self.rc.glm_dsa_mla_kv_spill_dir
        )
        if (stepped or k3_compressed_mla or glm53_compressed_mla
                or glm_dsa_spilled_mla):
            from .kv_cache import SteppedKVCache

            kv = SteppedKVCache(self.cfg.num_hidden_layers)
        elif (self.rc.qwen_fp8_kv_cache
                and self.cfg.model_type in ("qwen3_5", "qwen3_5_moe")):
            from .kv_cache import Fp8KVCache

            kv = Fp8KVCache(self.cfg.num_hidden_layers)
        else:
            kv = KVCache(self.cfg.num_hidden_layers)
        if (self.rc.gptoss_sliding_kv_window
                and self.cfg.model_type == "gpt_oss"
                and getattr(self.cfg, "sliding_window", 0)):
            bounded = kv.configure_sliding_windows(
                self.cfg.layer_types, int(self.cfg.sliding_window))
            if bounded:
                print(f"[engine] gpt-oss sliding KV window: {bounded} layers "
                      f"bounded to {int(self.cfg.sliding_window)} positions",
                      flush=True)
        if (self.rc.mla_compressed_kv
                and self.cfg.model_type in ("glm_moe_dsa", "glm5_next")):
            kv.compressed_mla = True
            kv.mla_absorbed = self.rc.mla_absorbed_decode
            if self.cfg.model_type == "glm5_next":
                kv.glm53_sparse_absorbed_mla = bool(
                    self.rc.glm53_sparse_absorbed_mla)
                kv.glm53_sparse_fused_attention = bool(
                    self.rc.glm53_sparse_fused_attention)
                kv.glm53_sparse_fused_kv_int8 = bool(
                    self.rc.glm53_sparse_fused_kv_int8)
            if not self._dsa_elided:  # F43 bounded mode provably never selects
                if self.cfg.model_type == "glm5_next":
                    from .glm5_next_dsa import GLM5NextDSAState

                    kv.dsa = GLM5NextDSAState(
                        self.cfg,
                        incremental_pool_cache=bool(
                            self.rc.glm53_incremental_dsa_pool),
                    )
                else:
                    from .glm_dsa import DSAState

                    kv.dsa = DSAState(
                        self.cfg,
                        key_tile_size=self.rc.glm_dsa_key_tile_size,
                        index_step_size=self.rc.glm_dsa_index_step_size,
                        index_preallocate=self.rc.glm_dsa_index_preallocate,
                        selection_query_tile_size=(
                            self.rc.glm_dsa_selection_query_tile_size),
                        selection_spill_dir=(
                            self.rc.glm_dsa_mla_kv_spill_dir),
                    )
            if glm_dsa_spilled_mla:
                kv.enable_latent_disk_spill(
                    self.rc.glm_dsa_mla_kv_spill_dir)
                kv.mla_absorbed_prefill = bool(
                    self.rc.glm_dsa_sparse_absorbed_mla)
        if k3_compressed_mla:
            # Explicit K3 candidate: retain only Moonshot's released
            # [c_kv | k_rope] latent, and use the capacity-stepped axis-1
            # cache so long prefill does not recopy the full prefix per tile.
            kv.compressed_mla = True
            kv.mla_absorbed = self.rc.kimi_k3_absorbed_mla
            kv.mla_absorbed_prefill = self.rc.kimi_k3_absorbed_mla
            kv.mla_absorbed_key_tile_size = int(
                self.rc.kimi_k3_mla_key_tile_size
            )
            if self.rc.kimi_k3_mla_kv_spill_dir:
                kv.enable_latent_disk_spill(
                    self.rc.kimi_k3_mla_kv_spill_dir)
        if self.cfg.model_type in _HYBRID_RECURRENT_MODEL_TYPES:
            # KDA's recurrent state is fixed-size and not token-indexed. Exact
            # endpoint/extension retention and durable restore carry this
            # companion cache alongside attention KV; arbitrary prefix trims
            # remain forbidden by the candidate-selection gate in generate().
            # F97: Jet-Nemotron's "jet" (JetBlock) layers use the SAME
            # KDAStateCache interface (.state/.set_state/.conv_history/
            # .set_conv_history) as Kimi Linear's KDA and Qwen3.5's DeltaNet
            # -- see runtime/jet_nemotron.py.
            attach_hybrid_recurrent_cache(
                kv,
                model_type=self.cfg.model_type,
                num_hidden_layers=self.cfg.num_hidden_layers,
                kda_spill_dir=(
                    self.rc.kimi_k3_kda_spill_dir
                    if self.cfg.model_type == "kimi_k3" else ""),
            )
        if self.cfg.model_type == "qwen4_exp":
            from .qwen4_exp_state import Qwen4ExpStateCache

            kv.qwen4_cache = Qwen4ExpStateCache(
                self.cfg.num_hidden_layers)
            kv.qwen4_cache.configure_qsa_pool_cache(
                self.rc.qwen4_qsa_pool_cache)
        return kv

    def _configure_restored_k3_spill(self, kv: KVCache) -> KVCache:
        """Give a durable K3 endpoint the live runtime's exact spill policy.

        The journal deliberately reconstructs a generic :class:`KVCache`: its
        immutable payload must not encode machine-local temporary directories.
        K3 serving, however, uses ``SteppedKVCache`` for compressed MLA and
        layer-stationary disk spill for both MLA and recurrent KDA state.  Apply
        those process-local mechanics after checksum validation and before the
        endpoint enters the hot-slot lifecycle.  This changes only placement;
        tensor dtype, shape, and bytes remain unchanged.
        """
        if self.cfg.model_type != "kimi_k3":
            return kv
        if self.rc.kimi_k3_mla_kv_spill_dir and kv.compressed_mla:
            from .kv_cache import SteppedKVCache

            kv = SteppedKVCache.from_cache(kv)
            # Durable payloads encode exact latent arrays, not process-local
            # execution flags. Reapply the same fingerprinted compressed-MLA
            # arithmetic selected by new_kv(); otherwise restart silently
            # expands all cached latents into per-head K/V even though the
            # cold process used absorbed MLA.
            kv.mla_absorbed = self.rc.kimi_k3_absorbed_mla
            kv.mla_absorbed_prefill = self.rc.kimi_k3_absorbed_mla
            kv.mla_absorbed_key_tile_size = int(
                self.rc.kimi_k3_mla_key_tile_size)
            if not kv.latent_spill_enabled:
                kv.enable_latent_disk_spill(
                    self.rc.kimi_k3_mla_kv_spill_dir)
        recurrent = getattr(kv, "kda_cache", None)
        if (
            recurrent is not None
            and self.rc.kimi_k3_kda_spill_dir
            and not recurrent.spill_enabled
        ):
            recurrent.enable_disk_spill(self.rc.kimi_k3_kda_spill_dir)
        return kv

    def _respill_completed_k3_state(self, kv: KVCache) -> dict[str, int]:
        """Release K3 endpoint arrays materialized by durable serialization.

        ``HotPromptKVPersistence.save`` must load every spilled array to produce
        one checksummed, restart-safe endpoint.  Leaving those arrays resident
        defeats K3's layer-stationary memory bound on the following request.
        Write the exact arrays back to the request's temporary spill tier and
        release their Metal owners.  Decode reloads each layer on demand.
        """
        counts = {"kda_layers": 0, "mla_layers": 0}
        if self.cfg.model_type != "kimi_k3":
            return counts
        recurrent = getattr(kv, "kda_cache", None)
        if recurrent is not None and getattr(recurrent, "spill_enabled", False):
            for layer in self.cfg.kda_layers:
                counts["kda_layers"] += int(recurrent.spill_layer(layer))
        if getattr(kv, "latent_spill_enabled", False):
            for layer in self.cfg.full_attn_layers:
                counts["mla_layers"] += int(kv.spill_latent_layer(layer))
        if counts["kda_layers"] or counts["mla_layers"]:
            mx.clear_cache()
        return counts

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
        direct_io_before = _direct_io_snapshot(self)
        qwen4_expert_before = (
            self.store.qwen4_fused_expert_snapshot()
            if self.cfg.model_type == "qwen4_exp" else None)
        qwen4_ple_before = (
            self._qwen4_ple_rows.telemetry()
            if self._qwen4_ple_rows is not None else None)
        if self.cfg.model_type == "qwen4_exp":
            self._qwen4_host_spool_stats = {
                "h2d_bytes": 0, "d2h_bytes": 0,
                "copy_seconds": 0.0, "peak_host_bytes": 0,
            }
        self._request_profiler = (
            telemetry.RequestProfiler(self.rc.execution_profile)
            if self.rc.execution_profile else None)
        if self.cfg.model_type in ("glm5_next", "glm_moe_dsa"):
            self._glm53_layer_stationary_stats = {}
        if self._dspark_tap_collector is not None:
            self._dspark_tap_collector.begin_attempt()
        if self._request_profiler is not None:
            self._request_profiler.set_phase("prefill")
        self._set_finegrained_fp8_direct_phase("prefill")
        sampling = sampling or SamplingParams()
        sampling.seed_rng()
        stop = stop or []
        # Text and vision own different KV implementations. A text request on a
        # vision-capable engine invalidates the retained multimodal prefix before
        # allocating its own state; image embeddings remain separately bounded.
        self._vision_prompt_cache = None
        self._glm53_vision_prompt_cache = None
        self._provisional = None  # F55 safety: a crashed spec round must not leave buffering on
        self._resident_fast_decode_sweeps = 0
        self._resident_fast_prefill_sweeps = 0
        self._disable_resident_fast_for_request = False
        self._resident_moe_sweeps = 0
        # The serving retry wrapper may restart only a prefill that has not
        # sampled or exposed any model output.  Once this becomes non-zero,
        # retrying would reuse a mutated grammar/RNG state and is forbidden.
        self._generation_sampled_tokens = 0
        self._qwen35_serial_verify_batched_mlp_layers = 0
        self._qwen35_serial_verify_batched_mlp_positions = 0
        self._qwen35_serial_verify_batched_mlp_s = 0.0
        self._qwen35_serial_verify_page_prepare_s = 0.0
        self._qwen35_serial_verify_cache_prepare_s = 0.0
        self._qwen35_serial_verify_page_reserve_s = 0.0
        self._qwen35_serial_verify_reserve_s = 0.0
        self._qwen35_serial_verify_weight_wait_s = 0.0
        self._qwen35_serial_verify_linear_layer_compute_s = 0.0
        self._qwen35_serial_verify_full_layer_compute_s = 0.0
        self._qwen35_serial_verify_head_s = 0.0
        self._qwen35_serial_verify_head_suspend_calls = 0
        self._qwen35_serial_verify_head_suspend_bytes = 0
        self._qwen35_serial_verify_head_suspend_active_released_bytes = 0
        self._qwen35_serial_verify_head_suspend_active_peak_bytes = 0
        self._qwen35_serial_verify_head_suspend_s = 0.0
        self._qwen35_serial_verify_head_restore_calls = 0
        self._qwen35_serial_verify_head_restore_successes = 0
        self._qwen35_serial_verify_head_restore_refusals = 0
        self._qwen35_serial_verify_head_restore_s = 0.0
        self._qwen35_lm_head_suspend_request_active = False
        self._qwen4_phase_head_suspend_calls = 0
        self._qwen4_phase_head_suspend_bytes = 0
        self._qwen4_phase_head_suspend_s = 0.0
        self._qwen4_phase_head_restore_calls = 0
        self._qwen4_phase_head_restore_successes = 0
        self._qwen4_phase_head_restore_refusals = 0
        self._qwen4_phase_head_restore_s = 0.0
        self._qwen4_serial_verify_head_suspend_calls = 0
        self._qwen4_serial_verify_head_suspend_bytes = 0
        self._qwen4_serial_verify_head_restore_trim_bytes = 0
        self._qwen4_serial_verify_exact_bf16_calls = 0
        self._qwen4_serial_verify_exact_bf16_rows = 0
        self._qwen4_serial_verify_exact_bf16_fallback_calls = 0
        self._qwen4_serial_verify_exact_bf16_fallback_reasons = {}
        self._true_peak_metal_bytes = mx.get_active_memory()  # see _note_true_peak
        if self.governor is not None:
            self.governor.reset_request_peak(self._true_peak_metal_bytes)
        self._expert_compute_batches = 0
        self._max_experts_per_compute_batch = 0
        self._adaptive_expert_batch_clamps = 0
        self._min_adaptive_expert_batch = 0
        self._expert_batch_prefetch_submitted = 0
        self._expert_batch_prefetch_wait_s = 0.0
        self._expert_batch_prefetch_hidden_s = 0.0
        self._expert_batch_prefetch_max_futures = 0
        self._expert_batch_prefetch_submitted_by_phase = {
            "prefill": 0, "decode": 0,
        }
        self._expert_batch_prefetch_wait_s_by_phase = {
            "prefill": 0.0, "decode": 0.0,
        }
        self._expert_batch_prefetch_hidden_s_by_phase = {
            "prefill": 0.0, "decode": 0.0,
        }
        self._expert_shared_overlap_layers = 0
        self.expert_route_overlap_trace = []
        self._expert_route_last_by_layer = {}
        self._expert_route_overlap_totals = {}
        self._qwen4_serial_verify_union_layers = 0
        self._qwen4_serial_verify_expert_slots = 0
        self._qwen4_serial_verify_union_experts = 0
        self._qwen4_serial_verify_expert_pages_avoided = 0
        self._qwen4_serial_verify_union_fetch_s = 0.0
        self._qwen4_serial_verify_page_prepare_s = 0.0
        self._qwen4_serial_verify_weight_wait_s = 0.0
        self._qwen4_serial_verify_reserve_s = 0.0
        self._qwen4_serial_verify_linear_compute_s = 0.0
        self._qwen4_serial_verify_full_compute_s = 0.0
        self._qwen4_serial_verify_linear_layers = 0
        self._qwen4_serial_verify_full_layers = 0
        self._qwen4_serial_verify_head_s = 0.0
        self._qwen4_serial_verify_pipelined_expert_layers = 0
        request_cache_before = _cache_io_snapshot(self)
        reranked_head = self._lm_head_w if self.rc.rerank_lm_head else None
        reranked_telemetry_before = (
            reranked_head.telemetry_snapshot()
            if callable(getattr(reranked_head, "telemetry_snapshot", None))
            else {})
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
            "hot_prompt_longer_disk_prefix_tokens": 0,
            "prompt_cache_lookup_s": 0.0,
            "hot_prompt_lookup_s": 0.0,
            "disk_prompt_lookup_s": 0.0,
            "prompt_tokenize_s": 0.0,
            "prompt_snapshot_write_s": 0.0,
            "postgen_snapshot_write_s": 0.0,
            "hot_prompt_kv_persist_write_s": 0.0,
            "hot_prompt_hybrid_prefix_snapshot_tokens": 0,
            "hot_prompt_boundary_matched_fork": 0,
            "hot_prompt_kv_gc_s": 0.0,
            "hot_prompt_kv_gc_removed": 0,
            "hot_prompt_kv_disk_hit": 0,
            "hot_prompt_kv_preloaded_disk_hit": 0,
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
            "hot_prompt_admission_positions": 0,
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
            "qwen_lossy_suffix_prefill_enabled": int(
                bool(self.rc.qwen_lossy_suffix_prefill_early_layers)),
            "qwen_lossy_suffix_prefill_used": 0,
            "qwen_lossy_suffix_prefill_early_layers": int(
                self.rc.qwen_lossy_suffix_prefill_early_layers),
            "qwen_lossy_suffix_prefill_prefix_tokens": int(
                self.rc.qwen_lossy_suffix_prefill_prefix_tokens),
            "qwen_lossy_suffix_prefill_tokens": int(
                self.rc.qwen_lossy_suffix_prefill_tokens),
            "qwen_compiled_delta_prefill": int(
                self.rc.qwen_compiled_delta_prefill),
            "qwen_native_fused_delta_prefill": int(
                self.rc.qwen_native_fused_delta_prefill),
            "qwen_chunked_delta_prefill": int(
                self.rc.qwen_chunked_delta_prefill),
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
            "suffix_decoding_kda_endpoint_capture_rounds": 0,
            "suffix_decoding_kda_endpoint_restore_rounds": 0,
            "suffix_decoding_kda_endpoint_retained_peak_bytes": 0,
            "suffix_decoding_kda_refeed_sweeps": 0,
            "suffix_decoding_kda_refeed_sweeps_saved": 0,
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
            "layer_stationary_endpoint_fused": 0,
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
            "reranked_lm_head_exact_resident_bytes": (
                self._reranked_lm_head_exact_resident_bytes),
            "reranked_lm_head_row_paged": int(bool(
                self._reranked_lm_head_source_fingerprint)),
            "reranked_lm_head_source_fingerprint": (
                self._reranked_lm_head_source_fingerprint),
            "reranked_lm_head_recall_probe_every": int(
                self.rc.rerank_lm_head_recall_probe_every),
            "reranked_lm_head_rank_capture": int(bool(
                self.rc.rerank_lm_head_rank_capture_path)),
            "reranked_lm_head_approx_fingerprint": (
                self._reranked_lm_head_approx_fingerprint),
            "expert_top_k_by_layer": list(self.cfg.expert_top_k_by_layer),
            "weight_integrity_mode": (
                self.store.integrity_mode
            ),
        }
        if self.cfg.model_type in ("qwen3_5", "qwen3_5_moe"):
            path_stats["qwen35_prefill_chunk_ceiling"] = int(
                self.rc.qwen35_prefill_chunk_ceiling)
            path_stats["qwen35_serial_verify_exact_page_admission"] = int(
                self.rc.qwen35_serial_verify_exact_page_admission)
            path_stats["qwen35_paged_online_attention"] = int(
                self.rc.qwen35_paged_online_attention)
            path_stats["qwen35_paged_online_tile_positions"] = int(
                self.rc.qwen35_paged_online_tile_positions)
            path_stats["qwen35_paged_online_page_native"] = int(
                self.rc.qwen35_paged_online_page_native)
            path_stats["qwen35_kv_page_positions"] = int(
                self.rc.kv_page_positions)
            path_stats["qwen35_serial_verify_batched_mlp"] = int(
                self.rc.qwen35_serial_verify_batched_mlp)
            path_stats["qwen35_serial_verify_suspend_lm_head"] = int(
                self.rc.qwen35_serial_verify_suspend_lm_head)
            path_stats[
                "qwen35_serial_verify_suspend_lm_head_min_prompt_tokens"
            ] = int(
                self.rc.qwen35_serial_verify_suspend_lm_head_min_prompt_tokens)
            path_stats["qwen35_serial_verify_head_suspend_calls"] = int(
                self._qwen35_serial_verify_head_suspend_calls)
            path_stats["qwen35_serial_verify_head_suspend_bytes"] = int(
                self._qwen35_serial_verify_head_suspend_bytes)
            path_stats[
                "qwen35_serial_verify_head_suspend_active_released_bytes"
            ] = int(
                self._qwen35_serial_verify_head_suspend_active_released_bytes)
            path_stats[
                "qwen35_serial_verify_head_suspend_active_peak_bytes"
            ] = int(
                self._qwen35_serial_verify_head_suspend_active_peak_bytes)
            path_stats["qwen35_serial_verify_head_suspend_s"] = float(
                self._qwen35_serial_verify_head_suspend_s)
            path_stats["qwen35_serial_verify_head_restore_calls"] = int(
                self._qwen35_serial_verify_head_restore_calls)
            path_stats["qwen35_serial_verify_head_restore_successes"] = int(
                self._qwen35_serial_verify_head_restore_successes)
            path_stats["qwen35_serial_verify_head_restore_refusals"] = int(
                self._qwen35_serial_verify_head_restore_refusals)
            path_stats["qwen35_serial_verify_head_restore_s"] = float(
                self._qwen35_serial_verify_head_restore_s)
        tokenize_t0 = time.perf_counter()
        prepared_ids = getattr(prompt, "token_ids", None)
        tokens = (list(prepared_ids) if prepared_ids is not None
                  else self.tokenizer.encode(prompt).ids)
        path_stats["prompt_tokenize_s"] = time.perf_counter() - tokenize_t0
        self._qwen35_lm_head_suspend_request_active = (
            qwen35_phase_head_request_active(
                self.rc.qwen35_serial_verify_suspend_lm_head,
                len(tokens),
                self.rc.qwen35_serial_verify_suspend_lm_head_min_prompt_tokens,
            ))
        path_stats[
            "qwen35_serial_verify_suspend_lm_head_request_active"
        ] = int(self._qwen35_lm_head_suspend_request_active)
        if self._qwen35_lm_head_suspend_request_active:
            self._suspend_qwen35_serial_verify_lm_head()
        elif (self.rc.qwen35_serial_verify_suspend_lm_head
                and self._qwen35_lm_head_pin_suspended):
            # Below the measured phase boundary, reproduce the established
            # pinned-head request shape exactly: materialize the dormant lease
            # before prefill and retain it through every verifier sweep.
            self._lm_head_weight()
        if self.cfg.model_type == "qwen4_exp" and self.rc.qwen4_phase_lm_head:
            # A prior request may have retained the phase pin for its complete
            # decode.  Drop it before any prompt trunk page or prompt state is
            # materialized.  The first endpoint-logit projection after prefill
            # restores the exact dormant lease automatically.
            self._suspend_qwen4_phase_lm_head()
        if self.cfg.model_type == "kimi_k3":
            if self.rc.kimi_k3_prefill_tile_policy == "prompt-length":
                (k3_tile_width, k3_dense_tile_size,
                 k3_schedule_bucket) = kimi_k3_prompt_prefill_schedule(
                    len(tokens),
                    policy=self.rc.kimi_k3_prefill_tile_policy,
                    long_context_tokens=(
                        self.rc.kimi_k3_prefill_long_context_tokens),
                    short_tile_width=(
                        self.rc.kimi_k3_prefill_short_tile_width),
                    long_tile_width=self._k3_prefill_long_tile_width,
                    short_dense_mlp_tile_size=(
                        self.rc.kimi_k3_dense_mlp_short_tile_size),
                    long_dense_mlp_tile_size=(
                        self._k3_dense_mlp_long_tile_size),
                )
                retry_ceiling = int(getattr(
                    self, "_hybrid_retry_chunk_ceiling", 0) or 0)
                if retry_ceiling:
                    k3_tile_width = min(k3_tile_width, retry_ceiling)
                self.rc.prefill_chunk_size = k3_tile_width
                self.rc.hot_prompt_kv_chunk_size = k3_tile_width
                self.rc.kimi_k3_dense_mlp_tile_size = k3_dense_tile_size
            else:
                k3_tile_width = int(self.rc.prefill_chunk_size)
                k3_dense_tile_size = int(
                    self.rc.kimi_k3_dense_mlp_tile_size)
                k3_schedule_bucket = "fixed"
            self._active_k3_prefill_schedule = (
                f"{k3_schedule_bucket}:prefill={k3_tile_width}:"
                f"dense={k3_dense_tile_size}"
            )
            path_stats["kimi_k3_prefill_tile_policy"] = (
                self.rc.kimi_k3_prefill_tile_policy)
            path_stats["kimi_k3_prefill_schedule"] = k3_schedule_bucket
            path_stats["kimi_k3_prefill_tile_width"] = k3_tile_width
            path_stats["kimi_k3_dense_mlp_tile_size"] = k3_dense_tile_size
            path_stats["kimi_k3_prefill_long_context_tokens"] = int(
                self.rc.kimi_k3_prefill_long_context_tokens)
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
            (
                (
                    self.rc.stepped_kv_threshold
                    and len(tokens) + max_tokens
                    > self.rc.stepped_kv_threshold
                )
                or (
                    self.rc.kimi_k3_compressed_mla
                    and self.cfg.model_type == "kimi_k3"
                )
            )
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
        exact_hidden = None
        precomputed_prompt_logits = None  # lossy PIC fills a complete prompt KV
        prompt_state_approximate = False
        reusable_watermark = 0
        persist_parent_chain: tuple[str, ...] = ()  # disk-segment parent for
        # this turn's save, if hot-kv persistence is enabled (see below)
        persist_parent_covered = 0  # exact token count `persist_parent_chain`
        boundary_segment_chain: tuple[str, ...] = ()
        # covers -- always equals `best_matched` when a match wins (true for
        # all three cases: endpoint/branch/repeat), 0 when cold. MUST be
        # passed to save() explicitly rather than re-derived, since a
        # "repeat" parent chain's last segment is not chunk-sized.
        # Recurrent state cannot be trimmed to an arbitrary common prefix.
        # It can, however, be transferred exactly at a complete retained
        # endpoint and extended with a suffix. Candidate selection below
        # limits hybrid models to those two no-trim cases.
        recurrent_exact_only = self.cfg.model_type in (
            "kimi_linear", "kimi_k3", "qwen3_5_moe", "qwen3_5",
            "qwen4_exp", "jet_nemotron", "glm5_next")
        hot_eligible = bool(
            self.rc.hot_prompt_kv
            and (not self.rc.max_kv_mb or self.rc.paged_kv_persist)
            and not force_adaptive_paged
            and not bool(getattr(prompt, "disable_hot_prompt_kv", False)))
        stable_boundary_positions = int(
            getattr(prompt, "stable_boundary_tokens", 0) or 0)
        resident_prompt_kv_bytes = self._project_dense_text_kv_bytes(
            len(tokens),
            stable_boundary_positions=(stable_boundary_positions or None))
        path_stats["prompt_kv_projected_bytes"] = int(
            resident_prompt_kv_bytes)
        path_stats["prompt_kv_projection"] = (
            "qwen35_mixed_depth"
            if (self.cfg.model_type in ("qwen3_5", "qwen3_5_moe")
                and self.rc.qwen_lossy_suffix_prefill_early_layers
                and len(tokens) > (
                    self.rc.qwen_lossy_suffix_prefill_prefix_tokens
                    + self.rc.qwen_lossy_suffix_prefill_tokens))
            else "uniform")
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
            best_exact_hidden = None
            best_reusable_watermark = 0
            best_lcp = 0
            best_needs_trim_to: int | None = None  # None = don't trim, else trim(N)
            best_case = None  # repeat | endpoint | extension | branch -- which arm won,
            # needed only to derive the correct disk-persistence parent chain below

            for idx, slot in enumerate(self._hot_prompt_slots):
                if (
                    self.cfg.model_type == "kimi_k3"
                    and not kimi_k3_prefill_schedule_compatible(
                        policy=self.rc.kimi_k3_prefill_tile_policy,
                        active_schedule=self._active_k3_prefill_schedule,
                        cached_schedule=getattr(
                            slot, "kimi_k3_prefill_schedule", ""),
                    )
                ):
                    # KDA/MLA endpoints are exact only for the schedule that
                    # constructed their lineage. Crossing the token threshold
                    # deliberately starts cold instead of mixing tile shapes.
                    continue
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
                            and slot.logits is not None
                            and (self.cfg.model_type != "qwen4_exp"
                                 or slot.exact_hidden is not None)):
                        candidate_matched = len(tokens)
                        candidate_exact_logits = slot.logits
                        candidate_exact_hidden = slot.exact_hidden
                        candidate_watermark = 0
                        candidate_trim_to = None
                        candidate_case = "endpoint"
                    elif (len(tokens) > len(slot.tokens)
                          and lcp == len(slot.tokens)):
                        candidate_matched = len(slot.tokens)
                        candidate_exact_logits = None
                        candidate_exact_hidden = None
                        candidate_watermark = 0
                        candidate_trim_to = None
                        candidate_case = "extension"
                    else:
                        continue
                    if candidate_matched > best_matched:
                        best_idx = idx
                        best_matched = candidate_matched
                        best_exact_logits = candidate_exact_logits
                        best_exact_hidden = candidate_exact_hidden
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
                    # F95: THIS candidate's own recorded chunk size, not the
                    # engine's current default -- different slots (different
                    # conversations) can legitimately have been built with
                    # different chunk sizes.
                    boundary = getattr(
                        slot, "chunk_size", 0) or self.rc.hot_prompt_kv_chunk_size
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
                    if (
                        self.cfg.model_type == "kimi_k3"
                        and not kimi_k3_prefill_schedule_compatible(
                            policy=self.rc.kimi_k3_prefill_tile_policy,
                            active_schedule=self._active_k3_prefill_schedule,
                            cached_schedule=getattr(
                                slot, "kimi_k3_prefill_schedule", ""),
                        )
                    ):
                        continue
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
                    # F95: this slot's own chunk size (tool_pic is not
                    # currently reachable for the models this varies for,
                    # but kept consistent with the matching loop above).
                    boundary = getattr(
                        slot, "chunk_size", 0) or self.rc.hot_prompt_kv_chunk_size
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

            preferred_disk_match = None
            if (kv is None
                    and best_idx is not None
                    and _prefer_longer_persisted_hybrid_prefix(
                        model_type=self.cfg.model_type,
                        best_case=best_case)
                    and self._hot_kv_persist is not None
                    and self._persisted_kv_restore_allowed()):
                # A recently used shorter boundary can coexist in RAM with a
                # strictly longer exact boundary on disk (for example after
                # switching between two branches and restarting). The hybrid
                # state cannot be trimmed, so ask the metadata journal whether
                # a longer candidate exists before consuming the RAM slot.
                # The threshold prevents payload hashing on the common path
                # where memory is already longest.
                preferred_disk_match = self._hot_kv_persist.find_best_match(
                    tokens,
                    self.rc.hot_prompt_kv_chunk_size,
                    cache_namespace=cache_namespace,
                    min_matched_exclusive=best_matched,
                )
                if preferred_disk_match is not None:
                    path_stats[
                        "hot_prompt_longer_disk_prefix_tokens"] = int(
                            preferred_disk_match["matched"])

            if kv is not None:
                pass
            elif (preferred_disk_match is None
                  and best_idx is not None and best_matched > 0):
                slot = self._hot_prompt_slots.pop(best_idx)  # consume: remove from the LRU
                if self._hybrid_chunk_size_applies():
                    continued_chunk = self._select_prefill_chunk_size(slot)
                    self.rc.prefill_chunk_size = continued_chunk
                    self.rc.hot_prompt_kv_chunk_size = continued_chunk
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
                exact_hidden = best_exact_hidden
                reusable_watermark = best_reusable_watermark
                path_stats["hot_prompt_lcp_tokens"] = best_lcp
                path_stats["prompt_cache_source"] = "memory"
                path_stats["hot_prompt_kv_preloaded_disk_hit"] = int(
                    bool(getattr(slot, "persisted_preload", False)))
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
                    disk_match = preferred_disk_match
                    if disk_match is None:
                        disk_match = self._hot_kv_persist.find_best_match(
                            tokens, self.rc.hot_prompt_kv_chunk_size,
                            cache_namespace=cache_namespace)
                    if disk_match is not None:
                        loaded = self._hot_kv_persist.load_matched_chain(
                            disk_match, self.cfg.num_hidden_layers)
                        if loaded is not None:
                            if len(loaded) == 4:
                                (loaded_tokens, loaded_kv,
                                 loaded_exact_logits,
                                 loaded_exact_hidden) = loaded
                            else:
                                (loaded_tokens, loaded_kv,
                                 loaded_exact_logits) = loaded
                                loaded_exact_hidden = None
                            kv = self._configure_restored_k3_spill(loaded_kv)
                            matched = disk_match["matched"]
                            exact_logits = loaded_exact_logits
                            exact_hidden = loaded_exact_hidden
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
            # Admission happens before the first `_sweep`, so establish the
            # retry chunk's phase margin here as well.  Without this, the final
            # token-at-a-time fallback still inherited the construction-time
            # 400 MB prefill pad and could be rejected before its first
            # size-one sweep had a chance to set the correct zero margin.
            # A durable/in-memory prefix may leave only a tiny scaffold suffix.
            # Size the phase reserve to work that will actually execute, not
            # the complete rendered prompt. Using ``len(tokens)-1`` here made
            # a 7-position restart inherit a 128-position transient estimate
            # and could reject a safe restore before its first layer.
            remaining_prefill_positions = max(1, len(tokens) - matched)
            admission_positions = self._prefill_admission_positions(
                remaining_prefill_positions)
            path_stats["hot_prompt_admission_positions"] = (
                admission_positions)
            position_transient = getattr(
                self, "_prefill_layer_transient_by_positions", {}
            ).get(admission_positions, 0)
            (self._layer_transient,
             self._layer_transient_margin) = _layer_transient_for_positions(
                 admission_positions, position_transient,
                 getattr(self, "_decode_layer_transient", 0))
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
                kv.online_attention = bool(
                    self.cfg.model_type in ("qwen3_5", "qwen3_5_moe")
                    and self.rc.qwen35_paged_online_attention)
                kv.online_attention_tile_positions = int(
                    self.rc.qwen35_paged_online_tile_positions)
                kv.online_attention_page_native = bool(
                    self.rc.qwen35_paged_online_page_native)
                kv.online_attention_pages_per_tile = int(
                    self.rc.qwen35_paged_online_tile_positions
                    // self.rc.kv_page_positions)
                attach_hybrid_recurrent_cache(
                    kv,
                    model_type=self.cfg.model_type,
                    num_hidden_layers=self.cfg.num_hidden_layers,
                    kda_spill_dir=(
                        self.rc.kimi_k3_kda_spill_dir
                        if self.cfg.model_type == "kimi_k3" else ""),
                )
            else:
                # F95: a genuinely NEW conversation -- no in-memory slot and
                # no disk match (persistence is off by default now anyway).
                # The slot created at the end of this request records
                # whatever value this becomes, so later turns of THIS
                # conversation stay pinned to it.
                if hot_eligible and self._hybrid_chunk_size_applies():
                    fresh_chunk = self._select_prefill_chunk_size(None)
                    self.rc.prefill_chunk_size = fresh_chunk
                    self.rc.hot_prompt_kv_chunk_size = fresh_chunk
                kv = self.new_kv(stepped=use_stepped_kv)
        self.last_kv = kv
        qwen4_state = getattr(kv, "qwen4_cache", None)
        if qwen4_state is not None:
            # Covers cold state, RAM endpoints, and disk-restored endpoints.
            # Preserve an exact derived cache on a RAM hit, but reset request
            # attribution. Durable state intentionally rebuilds from raw keys.
            qwen4_state.configure_qsa_pool_cache(
                self.rc.qwen4_qsa_pool_cache, reset_stats=True)
        if self.cfg.model_type in ("qwen3_5", "qwen3_5_moe"):
            self.rc.prefill_chunk_size = (
                self._apply_qwen35_prefill_chunk_ceiling(
                    self.rc.prefill_chunk_size))
        effective_prefill_chunk = int(self.rc.prefill_chunk_size or 0)
        path_stats["prefill_step_size"] = effective_prefill_chunk
        if self.cfg.model_type in ("qwen3_5", "qwen3_5_moe"):
            path_stats[
                "qwen35_prefill_chunk_selected"] = effective_prefill_chunk
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
        # F96: token count/state fork of this turn's stable boundary (this
        # conversation rendered WITHOUT its own trailing generation scaffold),
        # only ever populated below in the cold-sweep branch. Initialized here
        # so the post-generation slot-storage code can check it unconditionally
        # regardless of which of the three branches below actually ran.
        boundary_fork_tokens = 0
        boundary_fork_kv = None
        deferred_qwen_boundary_tokens = 0
        if (exact_logits is None
                and self.cfg.model_type == "deepseek_v4"
                and self._dsv4_prompt_reuse):
            # DeepSeek V4's prompt state is not a KVCache, so F37's store
            # cannot serialize it and load_longest_prefix always misses. The
            # whole state is small -- a 128-slot ring and a compressor carry
            # buffer per layer, about 20MB total -- so an exact-prompt
            # snapshot is cheap and skips the entire prefill sweep.
            snapshot = self._dsv4_snapshot_lookup(tokens)
            if snapshot is not None:
                self._dsv4_snapshot_restore(kv, snapshot)
                exact_logits = snapshot["logits"]
                path_stats["prompt_cache_source"] = "dsv4_state"
                path_stats["prompt_cache_prefix_tokens"] = len(tokens)
        if exact_logits is not None:
            logits = exact_logits  # exact hit: zero sweeps
            if exact_hidden is not None:
                self._h_window = exact_hidden
                self._h_last = exact_hidden[:, -1:, :]
            path_stats["prompt_cache_exact_hit"] = 1
        elif precomputed_prompt_logits is not None:
            # The selective PIC sweep already produced the complete prompt KV
            # and endpoint distribution during hot-cache planning.
            logits = precomputed_prompt_logits
        else:
            pos = matched
            if (self.cfg.model_type == "deepseek_v4"
                    and self.rc.dsv4_prefix_checkpoint_stride > 0
                    and pos == 0):
                resumed = self._dsv4_prefix_restore(kv, tokens)
                if resumed:
                    pos = resumed
                    path_stats["prompt_cache_source"] = "dsv4_prefix"
                    path_stats["prompt_cache_prefix_tokens"] = resumed
            # A layer-stationary sweep returns the hidden states for every
            # position it consumes.  When that sweep reaches the prompt
            # endpoint, retain its final hidden state for logits instead of
            # throwing it away and streaming the last prompt token through a
            # second complete weight sweep.  The ordinary chunk-major path
            # still needs its historical final-tail sweep because earlier
            # chunks intentionally discard their hidden states.
            layer_stationary_endpoint_x = None
            # Fork a retention checkpoint at this turn's stable boundary for
            # recurrent_exact_only models. The released chat template
            # re-renders any but the LATEST assistant turn without its own
            # generation scaffold once a further turn follows it, so a slot
            # retained past this point (the old behavior) can never match a
            # real second turn -- live-reproduced 2026-07-22 as a 100% hot-KV
            # miss rate on every qwen3_5/qwen3_5_moe conversation past its
            # first turn. Forking here instead means the retained slot's
            # tokens are exactly what ANY future continuation of this same
            # conversation is guaranteed to re-render byte-identically.
            if (recurrent_exact_only and hot_eligible
                    and (type(kv) in (KVCache, SteppedKVCache)
                         or self.rc.paged_kv_persist)):
                stable_boundary = int(
                    getattr(prompt, "stable_boundary_tokens", 0) or 0)
                matched_boundary_fork = _fork_matched_hybrid_stable_boundary(
                    kv,
                    matched_tokens=pos,
                    stable_boundary_tokens=stable_boundary,
                    prompt_tokens=len(tokens),
                )
                if matched_boundary_fork is not None:
                    # A restart/disk match can land exactly on the rendered
                    # scaffold-free boundary. Suffix prefill consumes and
                    # advances that matched state, so retain its cheap COW
                    # fork now just as the cold path does when it computes up
                    # to the boundary below. Without this equality arm, the
                    # one-slot RAM tier replaces a useful stable prefix with a
                    # post-generation recurrent endpoint that cannot be
                    # rewound for regenerate/next-turn requests; every later
                    # request is forced back to disk despite an exact match.
                    boundary_fork_kv = matched_boundary_fork
                    boundary_fork_tokens = stable_boundary
                    path_stats[
                        "hot_prompt_boundary_fork_tokens"] = stable_boundary
                    path_stats["hot_prompt_boundary_matched_fork"] = 1
                elif (self.rc.paged_kv_persist
                        and pos == stable_boundary
                        and persist_parent_covered == stable_boundary):
                    # A disk-restored stable prefix is already the immutable
                    # boundary checkpoint. Keep its chain as the endpoint
                    # parent; do not clone or rewrite any KV pages.
                    boundary_segment_chain = persist_parent_chain
                    path_stats[
                        "hot_prompt_boundary_fork_tokens"] = stable_boundary
                elif pos < stable_boundary < len(tokens):
                    # Chunk this mini-sweep exactly like the ordinary prefill
                    # loop below -- an unchunked sweep of an entire first-turn
                    # prompt here would bypass the same peak-memory bound
                    # every other prefill sweep in this method respects.
                    boundary_chunk = max(1, int(self.rc.prefill_chunk_size or (
                        stable_boundary - pos)))
                    bpos = pos
                    # F121: this boundary used to bypass F94/F35 completely.
                    # On a first agent turn it is usually almost the whole
                    # system+tool prompt, so the later layer-stationary branch
                    # saw only the tiny generation scaffold and truthfully
                    # reported itself eligible without ever running. That
                    # reintroduced chunk-major expert routing/refetch for the
                    # exact long-tool workload layer-stationary prefill was
                    # built to fix. Apply the same already-oracle-gated sweep
                    # to the boundary range when its ordinary eligibility
                    # conditions hold; the resulting endpoint state is the
                    # same state forked below.
                    # F128: kimi_k3 deliberately excluded -- see the same
                    # F128: _layer_stationary_kimi_k3_sweep now exists and
                    # is oracle-verified (tests/test_f128_k3_layer_
                    # stationary_oracle.py) against a scoped real-weight
                    # equivalence check, so kimi_k3 is included here too.
                    boundary_layer_stationary = (
                        self.rc.layer_stationary_prefill
                        and self.cfg.model_type in (
                            "qwen3_5", "qwen3_5_moe", "qwen4_exp", "gpt_oss",
                            "kimi_linear", "kimi_k3",
                            "glm_moe_dsa", "kimi_k25", "glm4_moe_lite",
                            "glm5_next")
                        and not self.rc.adaptive_chunk_size
                        and not (
                            self.rc.prefill_checkpoint_every
                            and kv_store is not None)
                        and not force_adaptive_paged
                    )
                    if self._request_profiler is not None:
                        self._request_profiler.note(
                            "hot_boundary layer_stationary "
                            f"eligible={int(boundary_layer_stationary)} "
                            f"positions={stable_boundary - pos}")
                    # Dense Qwen's generation scaffold can continue through
                    # the same resident layer weights as the stable prefix.
                    # Defer this one architecture to the complete-prompt
                    # layer-stationary sweep below; that sweep captures every
                    # layer's exact boundary endpoint before advancing the
                    # scaffold. Other recurrent architectures retain their
                    # already-gated separate boundary behavior.
                    fuse_qwen_boundary_scaffold = bool(
                        boundary_layer_stationary
                        and self.cfg.model_type in ("qwen3_5", "qwen3_5_moe")
                        and self.rc.qwen_fused_boundary_scaffold_prefill
                        and self.rc.prefill_chunk_size
                        and not self.rc.prefill_last_token_separate)
                    if fuse_qwen_boundary_scaffold:
                        boundary_fork_kv = fork_hybrid_kv_endpoint(kv)
                        boundary_fork_tokens = stable_boundary
                        deferred_qwen_boundary_tokens = stable_boundary
                        path_stats[
                            "hot_prompt_boundary_scaffold_fused"] = 1
                        path_stats[
                            "hot_prompt_boundary_fork_tokens"] = stable_boundary
                        if self._request_profiler is not None:
                            self._request_profiler.note(
                                "hot_boundary scaffold_fused=1 "
                                f"positions={len(tokens) - pos}")
                    elif boundary_layer_stationary:
                        bx = self._embed(list(tokens[pos:stable_boundary]))
                        if self.cfg.model_type == "kimi_linear":
                            bx = self._layer_stationary_kimi_linear_sweep(
                                bx, kv, offset=pos,
                                tile_width=boundary_chunk,
                                on_progress=on_progress)
                        elif self.cfg.model_type == "kimi_k3":
                            bx = self._layer_stationary_kimi_k3_sweep(
                                bx, kv, offset=pos,
                                tile_width=boundary_chunk,
                                on_progress=on_progress)
                        elif self.cfg.model_type == "gpt_oss":
                            bx = self._layer_stationary_gptoss_sweep(
                                bx, kv, offset=pos,
                                tile_width=boundary_chunk,
                                on_progress=on_progress)
                        elif self.cfg.model_type == "qwen4_exp":
                            bx = self._layer_stationary_qwen4_sweep(
                                bx, kv, offset=pos,
                                tile_width=boundary_chunk,
                                on_progress=on_progress)
                        elif self.cfg.model_type == "glm5_next":
                            bx = self._layer_stationary_glm5_next_sweep(
                                bx, kv, offset=pos,
                                tile_width=boundary_chunk,
                                on_progress=on_progress)
                        elif self.cfg.model_type in (
                                "glm_moe_dsa", "kimi_k25", "glm4_moe_lite"):
                            bx = self._layer_stationary_glm_sweep(
                                bx, kv, offset=pos,
                                tile_width=boundary_chunk,
                                on_progress=on_progress)
                        else:
                            use_lossy_suffix = (
                                pos == 0
                                and bool(
                                    self.rc
                                    .qwen_lossy_suffix_prefill_early_layers)
                                and stable_boundary - pos
                                > (
                                    self.rc
                                    .qwen_lossy_suffix_prefill_prefix_tokens
                                    + self.rc
                                    .qwen_lossy_suffix_prefill_tokens)
                            )
                            if use_lossy_suffix:
                                bx = self._qwen35_lossy_suffix_prefill_sweep(
                                    bx, kv, offset=pos,
                                    tile_width=boundary_chunk,
                                    on_progress=on_progress)
                                prompt_state_approximate = True
                                path_stats[
                                    "qwen_lossy_suffix_prefill_used"] = 1
                            else:
                                bx = self._layer_stationary_qwen35_sweep(
                                    bx, kv, offset=pos,
                                    tile_width=boundary_chunk,
                                    on_progress=on_progress)
                        del bx
                        bpos = stable_boundary
                        path_stats["hot_prompt_boundary_layer_stationary"] = 1
                    else:
                        while bpos < stable_boundary:
                            bend = min(bpos + boundary_chunk, stable_boundary)
                            bx = self._embed(list(tokens[bpos:bend]))
                            bx = self._sweep(
                                bx, kv, offset=bpos,
                                final_mlp_last_only=self.rc.final_dead_token_elim)
                            bpos = bend
                            if on_progress is not None:
                                on_progress({
                                    "phase": "prefill",
                                    "completed_tokens": bpos,
                                    "total_tokens": len(tokens),
                                    "cache_source": path_stats[
                                        "prompt_cache_source"],
                                })
                    if not fuse_qwen_boundary_scaffold:
                        path_stats["hot_prompt_boundary_prefill_chunks"] = (
                            -(-(stable_boundary - pos) // boundary_chunk))
                        if self.rc.paged_kv_persist:
                            recurrent = getattr(kv, "kda_cache", None)
                            if recurrent is None or self._hot_kv_persist is None:
                                raise RuntimeError(
                                    "paged Qwen boundary is missing durable "
                                    "recurrent state")
                            recurrent.synchronize()
                            boundary_segment_chain = self._hot_kv_persist.save(
                                parent_chain=persist_parent_chain,
                                parent_covered=persist_parent_covered,
                                tokens=tuple(tokens[:stable_boundary]),
                                kv=kv,
                                logits=None,
                                prompt_logits=None,
                                prompt_length=stable_boundary,
                                reusable_prefix=stable_boundary,
                                approximate=prompt_state_approximate,
                                tool_capsules=(),
                                cache_namespace=cache_namespace,
                                checkpoint_kind="stable_prefix",
                            )
                            persist_parent_chain = boundary_segment_chain
                            persist_parent_covered = stable_boundary
                            path_stats[
                                "hot_prompt_hybrid_prefix_snapshot_tokens"] = (
                                    stable_boundary)
                        else:
                            boundary_fork_kv = fork_hybrid_kv_endpoint(kv)
                            boundary_fork_tokens = stable_boundary
                        pos = stable_boundary
                        path_stats[
                            "hot_prompt_boundary_fork_tokens"] = stable_boundary
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
            if (chunk and self.rc.adaptive_chunk_size
                    and not getattr(self, "_adaptive_chunk_pinned_after_retry", False)):
                from .adaptive_chunk import AdaptiveChunkController

                adaptive_dynamic_ceiling = self.rc.adaptive_chunk_safe_bytes == 0
                adaptive_safe_bytes = (
                    self.governor.current_ceiling()
                    if adaptive_dynamic_ceiling
                    else self.rc.adaptive_chunk_safe_bytes
                )
                adaptive = AdaptiveChunkController(
                    safe_bytes=adaptive_safe_bytes, initial_chunk=chunk,
                    escalate_growth_cap=self.rc.adaptive_chunk_escalate_growth_cap,
                    worst_case_expert_bytes_per_token=(
                        self.cfg.num_experts_per_tok * self._expert_fetch_page_bytes),
                    max_expert_fetch_bytes=(
                        self.cfg.num_experts * self._expert_fetch_page_bytes))
                path_stats["adaptive_chunk_events"] = adaptive.events
                path_stats["adaptive_chunk_dynamic_ceiling"] = int(
                    adaptive_dynamic_ceiling)
                path_stats["adaptive_chunk_safe_bytes_min"] = adaptive_safe_bytes
                path_stats["adaptive_chunk_safe_bytes_max"] = adaptive_safe_bytes
            pressure_chunk_events: list[dict] = []
            observed_swap_pressure_events = int(getattr(
                self.governor, "swap_pressure_events", 0) or 0)
            path_stats["pressure_chunk_events"] = pressure_chunk_events
            # F94: layer-major fast path, opt-in (rc.layer_stationary_prefill)
            # and narrowly scoped -- only when none of the features below that
            # need PER-CHUNK control (adaptive resizing, mid-prefill
            # checkpoint saves, paged-KV forced chunking) are in play for THIS
            # request. Anything more complex falls through to the existing,
            # already-safe chunk-major loop unchanged. Eligibility is checked
            # fresh every call (never cached), so a request that later needs
            # one of those features is simply never routed here.
            # F128: kimi_k3 deliberately excluded -- see the
            # F128: _layer_stationary_kimi_k3_sweep now exists and is
            # oracle-verified against a scoped real-weight equivalence
            # check, so kimi_k3 is included here too.
            layer_stationary_eligible = (
                bool(chunk)
                and self.rc.layer_stationary_prefill
                and self.cfg.model_type in (
                    "qwen3_5", "qwen3_5_moe", "qwen4_exp", "gpt_oss",
                    "kimi_linear", "kimi_k3", "deepseek_v4",
                    "glm_moe_dsa", "kimi_k25", "glm4_moe_lite",
                    "glm5_next")
                and adaptive is None
                and not (ckpt and kv_store is not None)
                and not force_adaptive_paged
            )
            if self._request_profiler is not None:
                blockers = []
                if not chunk:
                    blockers.append("no_chunk")
                if not self.rc.layer_stationary_prefill:
                    blockers.append("disabled")
                if self.cfg.model_type not in (
                        "qwen3_5", "qwen3_5_moe", "qwen4_exp", "gpt_oss",
                        "kimi_linear", "kimi_k3", "deepseek_v4",
                        "glm_moe_dsa", "kimi_k25", "glm4_moe_lite",
                        "glm5_next"):
                    blockers.append("architecture")
                if adaptive is not None:
                    blockers.append("adaptive_chunk")
                if ckpt and kv_store is not None:
                    blockers.append("checkpoint_store")
                if force_adaptive_paged:
                    blockers.append("paged_kv")
                self._request_profiler.note(
                    "layer_stationary "
                    f"configured={int(self.rc.layer_stationary_prefill)} "
                    f"eligible={int(layer_stationary_eligible)} "
                    f"chunk={int(chunk or 0)} "
                    f"blockers={','.join(blockers) if blockers else 'none'}")
            if deferred_qwen_boundary_tokens and not layer_stationary_eligible:
                raise RuntimeError(
                    "deferred Qwen boundary lost layer-stationary eligibility")
            if layer_stationary_eligible:
                prefill_limit = (
                    len(tokens) - 1
                    if self.rc.prefill_last_token_separate and len(tokens) > 1
                    else len(tokens)
                )
                # Replicate the ordinary loop's own chunk-boundary arithmetic
                # (below) purely to find where IT would stop -- chunk is
                # constant here (adaptive is None, no mid-loop pressure-driven
                # resizing is attempted in this fast path), so this is exact,
                # not an approximation. The true final tail is always left for
                # the unchanged code after this whole if/else, exactly as the
                # ordinary loop already leaves it.
                stop_before = pos
                while stop_before < prefill_limit:
                    end = min(stop_before + chunk, prefill_limit)
                    stop_before = end
                if stop_before > pos:
                    # F96 hot-KV bookkeeping: the ordinary loop advances
                    # reusable_watermark by exactly `chunk` on every iteration
                    # where chunk_start == reusable_watermark (i.e. as long as
                    # watermark tracking hasn't already fallen behind `pos`).
                    # chunk is constant here, so by induction the same holds
                    # for every full-chunk step this fast path takes -- only
                    # replicate it when that continuity actually holds at the
                    # start, exactly matching what the ordinary loop would
                    # have done, not a separate/weaker approximation of it.
                    watermark_continuous = (
                        hot_eligible and reusable_watermark == pos)
                    xc = self._embed(list(tokens[pos:stop_before]))
                    if self.cfg.model_type == "deepseek_v4":
                        xc = self._layer_stationary_deepseek_v4_sweep(
                            xc, kv, offset=pos, tile_width=chunk,
                            on_progress=on_progress)
                    elif self.cfg.model_type == "kimi_linear":
                        xc = self._layer_stationary_kimi_linear_sweep(
                            xc, kv, offset=pos, tile_width=chunk,
                            on_progress=on_progress)
                    elif self.cfg.model_type == "kimi_k3":
                        xc = self._layer_stationary_kimi_k3_sweep(
                            xc, kv, offset=pos, tile_width=chunk,
                            on_progress=on_progress)
                    elif self.cfg.model_type == "gpt_oss":
                        xc = self._layer_stationary_gptoss_sweep(
                            xc, kv, offset=pos, tile_width=chunk,
                            on_progress=on_progress)
                    elif self.cfg.model_type == "qwen4_exp":
                        xc = self._layer_stationary_qwen4_sweep(
                            xc, kv, offset=pos, tile_width=chunk,
                            on_progress=on_progress)
                    elif self.cfg.model_type == "glm5_next":
                        xc = self._layer_stationary_glm5_next_sweep(
                            xc, kv, offset=pos, tile_width=chunk,
                            on_progress=on_progress)
                    elif self.cfg.model_type in (
                            "glm_moe_dsa", "kimi_k25", "glm4_moe_lite"):
                        xc = self._layer_stationary_glm_sweep(
                            xc, kv, offset=pos, tile_width=chunk,
                            on_progress=on_progress)
                    else:
                        use_lossy_suffix = (
                            pos == 0
                            and bool(
                                self.rc
                                .qwen_lossy_suffix_prefill_early_layers)
                            and stop_before - pos
                            > (
                                self.rc
                                .qwen_lossy_suffix_prefill_prefix_tokens
                                + self.rc
                                .qwen_lossy_suffix_prefill_tokens)
                        )
                        if use_lossy_suffix:
                            xc = self._qwen35_lossy_suffix_prefill_sweep(
                                xc, kv, offset=pos, tile_width=chunk,
                                on_progress=on_progress,
                                stable_boundary_tokens=(
                                    deferred_qwen_boundary_tokens - pos
                                    if deferred_qwen_boundary_tokens else None),
                                boundary_fork_kv=(
                                    boundary_fork_kv
                                    if deferred_qwen_boundary_tokens else None))
                            prompt_state_approximate = True
                            path_stats[
                                "qwen_lossy_suffix_prefill_used"] = 1
                        else:
                            xc = self._layer_stationary_qwen35_sweep(
                                xc, kv, offset=pos, tile_width=chunk,
                                on_progress=on_progress,
                                boundary_fork_at=(
                                    deferred_qwen_boundary_tokens - pos
                                    if deferred_qwen_boundary_tokens else None),
                                boundary_fork_kv=(
                                    boundary_fork_kv
                                    if deferred_qwen_boundary_tokens else None))
                    if stop_before == len(tokens):
                        layer_stationary_endpoint_x = xc
                        path_stats["layer_stationary_endpoint_fused"] = 1
                    else:
                        del xc
                    if self.rc.max_kv_mb or force_adaptive_paged:
                        mx.clear_cache()
                    path_stats["prefill_chunks"] += -(-(stop_before - pos) // chunk)
                    if watermark_continuous:
                        reusable_watermark = pos + (
                            (stop_before - pos) // chunk) * chunk
                        path_stats["hot_prompt_reusable_prefix_tokens"] = (
                            reusable_watermark)
                    pos = stop_before
            elif chunk:
                prefill_limit = (
                    len(tokens) - 1
                    if self.rc.prefill_last_token_separate and len(tokens) > 1
                    else len(tokens)
                )
                while pos < prefill_limit:
                    chunk_start = pos
                    gov = self.governor
                    live_swap_events = int(getattr(
                        gov, "swap_pressure_events", 0) or 0)
                    if (live_swap_events > observed_swap_pressure_events
                            and adaptive is None
                            and self._memory_prefill_retry_applies()
                            and chunk > 1):
                        old_chunk = chunk
                        chunk = next(
                            (candidate for candidate in (128, 32, 8, 1)
                             if candidate < old_chunk),
                            1,
                        )
                        self.rc.prefill_chunk_size = chunk
                        self.rc.hot_prompt_kv_chunk_size = chunk
                        # Mixed chunk boundaries remain an exact sequential
                        # prefill schedule, but this request cannot safely be
                        # advertised as a fixed-boundary hot-cache lineage.
                        hot_eligible = False
                        pressure_chunk_events.append({
                            "position": pos,
                            "from": old_chunk,
                            "to": chunk,
                            "swap_pressure_event": live_swap_events,
                        })
                        print(
                            f"[prefill] live swap pressure at position={pos}; "
                            f"chunk {old_chunk}->{chunk} and disable hot retain",
                            flush=True,
                        )
                    observed_swap_pressure_events = live_swap_events
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
                        expert_misses_before = self.expert_misses
                    xc = self._embed(list(tokens[pos:end]))
                    xc = self._sweep(xc, kv, offset=pos,
                                     final_mlp_last_only=self.rc.final_dead_token_elim)
                    path_stats["prefill_chunks"] += 1
                    if adaptive is not None:
                        gov_event = gov is not None and (
                            gov.shrinks > shrinks_before or gov.reservations > reservations_before)
                        # This chunk's own real expert-page-miss cost --
                        # residualized out of the growth-fit's training data
                        # below (see AdaptiveChunkController.observe's
                        # docstring for why this confound otherwise collapses
                        # the chunk size on a small-expert-pool model).
                        chunk_expert_misses = self.expert_misses - expert_misses_before
                        adaptive.observe(
                            chunk_size=end - pos, peak=self._chunk_peak_metal_bytes,
                            active_before=active_before, kv_before=kv_before,
                            governor_event=gov_event,
                            expert_fetch_bytes=(
                                chunk_expert_misses * self._expert_fetch_page_bytes))
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
            if layer_stationary_endpoint_x is not None:
                x = layer_stationary_endpoint_x
            else:
                x = self._embed(list(tokens[pos:]))
                # F36 applies here because generate() consumes only the last
                # position; forward_tokens (speculative verify) must NOT use
                # it — it needs logits and trunk states at every fed position.
                x = self._sweep(
                    x, kv, offset=pos,
                    final_mlp_last_only=self.rc.final_dead_token_elim)
            if (hot_eligible
                    and pos == reusable_watermark
                    and len(tokens) - pos == self.rc.hot_prompt_kv_chunk_size):
                reusable_watermark = len(tokens)
                path_stats["hot_prompt_reusable_prefix_tokens"] = reusable_watermark
            self._h_window = x
            self._h_last = x[:, -1:, :]
            logits = self._final_logits(x)
        if (
            hot_eligible
            and exact_logits is None
            and prompt_state_approximate
            and self.rc.qwen_mixed_depth_endpoint_persist
            and self._hot_kv_persist is not None
            and getattr(
                self._hot_kv_persist,
                "supports_prompt_endpoint_snapshot",
                False,
            )
        ):
            endpoint_write_t0 = time.perf_counter()
            endpoint_chain = self._hot_kv_persist.save_prompt_endpoint(
                tokens,
                kv,
                logits,
                self._h_last,
                approximate=True,
                cache_namespace=cache_namespace,
            )
            path_stats["hot_prompt_endpoint_snapshot_tokens"] = len(tokens)
            path_stats["hot_prompt_endpoint_snapshot_write_s"] = (
                time.perf_counter() - endpoint_write_t0)
            path_stats["hot_prompt_endpoint_snapshot_id"] = (
                endpoint_chain[-1] if endpoint_chain else "")
        sampled_logits = self._constraint_logits(logits, constraint)
        next_tok = sample(sampled_logits, sampling, history=tokens)
        if constraint is not None:
            constraint.accept_token(next_tok)
        self._generation_sampled_tokens = 1
        grammar_completed = bool(
            constraint is not None and constraint.completed)
        prompt_endpoint_logits = logits
        prefill_cache_after = _cache_io_snapshot(self)
        prefill_s = (time.perf_counter() - t0
                     + path_stats["tool_pic_prefill_s"])
        self._set_finegrained_fp8_direct_phase("decode")
        if self._request_profiler is not None:
            self._request_profiler.set_phase("decode")
        if (self.cfg.model_type == "deepseek_v4"
                and self._dsv4_prompt_reuse
                and path_stats["prompt_cache_source"] != "dsv4_state"):
            self._dsv4_snapshot_store(tokens, kv, logits)
        if self.cfg.model_type == "deepseek_v4":
            stored = self._dsv4_prefix_store(tokens)
            if stored:
                path_stats["prefill_checkpoints_saved"] = 1
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
            # F102: this path calls mx.argmax directly, bypassing sample()
            # entirely -- a non-default repetition_penalty would otherwise
            # be silently ignored even though is_greedy stays True (the
            # penalty doesn't change the sampling STRATEGY, just the logits
            # it strategy operates on).
            and sampling.repetition_penalty == 1.0
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
            path_stats["suffix_decoding_kda_endpoint_capture_rounds"] = (
                suffix_stats.kda_endpoint_capture_rounds
            )
            path_stats["suffix_decoding_kda_endpoint_restore_rounds"] = (
                suffix_stats.kda_endpoint_restore_rounds
            )
            path_stats[
                "suffix_decoding_kda_endpoint_retained_peak_bytes"
            ] = suffix_stats.kda_endpoint_retained_peak_bytes
            path_stats["suffix_decoding_kda_refeed_sweeps"] = (
                suffix_stats.kda_refeed_sweeps
            )
            path_stats["suffix_decoding_kda_refeed_sweeps_saved"] = (
                suffix_stats.kda_refeed_sweeps_saved
            )
        elif can_pipeline:
            # Submit token N+1 before waiting for token N. The lazy token itself
            # is a valid gather index for the following graph, so CPU graph
            # construction overlaps Metal execution instead of leaving a bubble
            # at every greedy boundary. This is the largest measured dense Q4
            # side-quest win after removing per-layer synchronization.
            decode_t0 = time.perf_counter()
            boundary = mx.get_active_memory()
            misses_before = self.cache.stats.misses
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
                self._generation_sampled_tokens = len(generated)
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
            # See the plain per-token loop's identical guard below for why:
            # a mid-stream cache miss (e.g. a budget eviction while
            # pipelined) would fold one-time fetch/quantize scratch into
            # this engine-lifetime ratchet otherwise.
            if self.cache.stats.misses == misses_before:
                self._token_transient = max(
                    self._token_transient,
                    _resident_adjusted_transient(
                        boundary, mx.get_active_memory(), mx.get_peak_memory()))
            self._note_true_peak()
            decode_elapsed = time.perf_counter() - decode_t0
            tok_times = [decode_elapsed / pipelined_decode_steps] * pipelined_decode_steps
        elif stop_text is None:
            # Grammar fast-forward (rc.grammar_fast_forward, 2026-07-23):
            # `pending` holds sampled/forced tokens not yet fed to the model.
            # The plain path keeps it at exactly [next_tok] (identical
            # behavior); fast-forward extends it with grammar-FORCED tokens
            # (positions where the mask allows exactly one id -- masked
            # argmax is then deterministic, so no per-token sweep is needed
            # to decide them) and the next iteration feeds the whole run in
            # ONE multi-position sweep. Termination invariants preserved
            # everywhere: kv always ends holding prompt+generated[:-1] and
            # `logits` the distribution that predicted generated[-1] (the
            # postgen kv_store.save below depends on both).
            pending = [next_tok]

            def _grammar_fast_forward() -> bool:
                """Emit forced tokens; True if generation must terminate."""
                nonlocal stop_text, matched_stop_sequence, grammar_completed
                nonlocal pending, logits
                forced = constraint.forced_run(
                    max(0, max_tokens - len(generated)),
                    encode=(
                        (lambda text: self.tokenizer.encode(text).ids)
                        if self.rc.grammar_jump_forward_lossy else None))
                if not forced:
                    return False
                grammar_completed = bool(constraint.completed)
                path_stats["grammar_fast_forward_tokens"] = path_stats.get(
                    "grammar_fast_forward_tokens", 0) + len(forced)
                terminal = False
                committed: list[int] = []
                for tok in forced:
                    generated.append(tok)
                    committed.append(tok)
                    if stop:
                        decoded = self.tokenizer.decode(generated)
                        match = _stop_match(decoded)
                        if match is not None:
                            cut, _order, matched_stop_sequence = match
                            stop_text = decoded[:cut]
                            terminal = True
                            break  # never streamed, matching the plain path
                    if stream_decoder is not None:
                        delta = stream_decoder.push(generated)
                        if delta:
                            on_token(delta)
                    if (tok in self.cfg.eos_token_ids
                            or len(generated) >= max_tokens):
                        terminal = True
                        break
                self._generation_sampled_tokens = len(generated)
                if terminal or grammar_completed:
                    # Final-token-never-fed contract: feed everything up to
                    # generated[-2] now so kv/logits land in exactly the
                    # state the plain per-token loop leaves them in.
                    feed = pending + committed[:-1]
                    if feed:
                        xf = self._embed(feed)
                        xf = self._sweep(xf, kv, offset=kv.offset)
                        logits = self._final_logits(xf)
                        mx.eval(logits)
                    pending = [generated[-1]]
                    return True
                pending = pending + committed
                return False

            ff_active = (
                constraint is not None
                and (self.rc.grammar_fast_forward
                     or self.rc.grammar_jump_forward_lossy))
            ff_terminated = False
            if ff_active and not grammar_completed:
                # The endpoint-sampled first token may itself open a forced
                # span (e.g. "<tool_call>" scaffolding) before any decode
                # iteration runs.
                ff_terminated = _grammar_fast_forward()
            if not ff_terminated:
                while len(generated) < max_tokens:
                    if grammar_completed or next_tok in self.cfg.eos_token_ids:
                        break
                    t0 = time.perf_counter()
                    # F42: the real overshoot is the WHOLE-TOKEN transient
                    # (measured at the greedy() sync point, not inside any one
                    # layer) — learn it and reserve before the next token so
                    # the ceiling is never crossed.
                    boundary = mx.get_active_memory()
                    misses_before = self.cache.stats.misses
                    reserve_decode_step(kv)
                    mx.reset_peak_memory()
                    x = self._embed(pending)
                    x = self._sweep(x, kv, offset=kv.offset)
                    logits = self._final_logits(x)
                    sampled_logits = self._constraint_logits(
                        logits, constraint, hidden=x)
                    next_tok = sample(
                        sampled_logits, sampling,
                        history=tokens + generated
                        if sampling.repetition_penalty != 1.0 else None)
                    if constraint is not None:
                        constraint.accept_token(next_tok)
                        grammar_completed = bool(constraint.completed)
                    # 2026-07-23: a cache MISS during this step means at least
                    # one layer was fetched fresh (quantize-on-load configs:
                    # real bf16->Q4 conversion scratch, not just an I/O wait) --
                    # a one-time cost, not the steady-state per-token transient
                    # this ratchet exists to learn. self._token_transient never
                    # resets across requests (engine-lifetime), so folding a
                    # polluted first-decode-token measurement in here silently
                    # disabled F99's resident fast path (reserve_decode_step's
                    # own check) for the rest of THIS engine's life, not just
                    # this one request -- live-caught investigating why F99
                    # never engaged on quantize-on-load configs.
                    if self.cache.stats.misses == misses_before:
                        self._token_transient = max(
                            self._token_transient,
                            _resident_adjusted_transient(
                                boundary, mx.get_active_memory(),
                                mx.get_peak_memory()))
                    self._note_true_peak()
                    tok_times.append(time.perf_counter() - t0)
                    generated.append(next_tok)
                    self._generation_sampled_tokens = len(generated)
                    pending = [next_tok]
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
                    if ff_active and not grammar_completed:
                        if _grammar_fast_forward():
                            break
                if len(pending) > 1:
                    # Loop exhausted max_tokens with an unfed fast-forward
                    # run still pending: restore the kv/logits endpoint
                    # contract (see _grammar_fast_forward's terminal branch).
                    xf = self._embed(pending[:-1])
                    xf = self._sweep(xf, kv, offset=kv.offset)
                    logits = self._final_logits(xf)
                    mx.eval(logits)

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
        if self.cfg.model_type == "glm_moe_dsa":
            path_stats["glm_dsa_dense_mlp_tile_size"] = int(
                self.rc.glm_dsa_dense_mlp_tile_size)
            path_stats["glm_dsa_index_preallocate"] = int(
                self.rc.glm_dsa_index_preallocate)
        if self.cfg.model_type in ("glm_moe_dsa", "glm5_next"):
            path_stats["glm53_native_fp8_dequant"] = int(
                bool(getattr(self.store, "native_glm53_fp8_dequant", False)))
            path_stats["glm53_fp8_direct_qmv"] = int(
                bool(getattr(self.store, "glm53_fp8_direct_qmv", False)))
            path_stats["glm53_fp8_direct_qmv_decode_only"] = int(bool(
                getattr(
                    self.store, "glm53_fp8_direct_qmv_decode_only", False)))
            path_stats["glm53_native_fp8_prefetch"] = int(
                bool(getattr(self.store, "native_glm53_fp8_prefetch", True)))
        if self.cfg.model_type == "qwen4_exp":
            path_stats["qwen4_native_fp8_dequant"] = int(
                bool(getattr(self.store, "native_glm53_fp8_dequant", False)))
            path_stats["qwen4_per_expert_fp8"] = int(
                getattr(self.store, "qwen4_expert_layout", "")
                == "per-expert-fp8")
            path_stats["qwen4_fp8_direct_qmv"] = int(
                bool(getattr(self.store, "qwen4_fp8_direct_qmv", False)))
            path_stats["qwen4_fp8_direct_qmv_decode_only"] = int(bool(
                getattr(
                    self.store, "qwen4_fp8_direct_qmv_decode_only", False)))
        dsa_state = getattr(kv, "dsa", None)
        if dsa_state is not None:
            path_stats["dsa_observations"] = dsa_state.stats["observations"]
            path_stats["dsa_sparse_selects"] = dsa_state.stats["sparse_selects"]
            path_stats["dsa_shared_reuses"] = dsa_state.stats["shared_reuses"]
            path_stats["dsa_score_tiles"] = int(
                dsa_state.stats.get("score_tiles", 0))
            path_stats["dsa_score_syncs"] = int(
                dsa_state.stats.get("score_syncs", 0))
            path_stats["dsa_score_candidate_id_sorts_avoided"] = int(
                dsa_state.stats.get(
                    "score_candidate_id_sorts_avoided", 0))
            path_stats["dsa_score_final_sorts_avoided"] = int(
                dsa_state.stats.get("score_final_sorts_avoided", 0))
            path_stats["dsa_multi_query_selects"] = int(
                dsa_state.stats.get("multi_query_selects", 0))
            path_stats["dsa_selection_ranges_peak"] = int(
                dsa_state.stats.get("selection_ranges_peak", 0))
            path_stats["dsa_selection_bytes_peak"] = int(
                dsa_state.stats.get("selection_bytes_peak", 0))
            path_stats["dsa_preselection_groups"] = int(
                dsa_state.stats.get("preselection_groups", 0))
            path_stats["dsa_preselection_queries"] = int(
                dsa_state.stats.get("preselection_queries", 0))
            path_stats["dsa_preselection_attention_ranges"] = int(
                dsa_state.stats.get("preselection_attention_ranges", 0))
            path_stats["dsa_index_capacity_grows"] = int(
                dsa_state.stats.get("index_capacity_grows", 0))
            path_stats["dsa_index_capacity_preallocations"] = int(
                dsa_state.stats.get("index_capacity_preallocations", 0))
            path_stats["dsa_index_capacity_preallocated_rows"] = int(
                dsa_state.stats.get("index_capacity_preallocated_rows", 0))
            path_stats["dsa_index_rows_copied"] = int(
                dsa_state.stats.get("index_rows_copied", 0))
            path_stats["dsa_index_rows_appended"] = int(
                dsa_state.stats.get("index_rows_appended", 0))
            path_stats["dsa_index_capacity_rows_peak"] = int(
                dsa_state.stats.get("index_capacity_rows_peak", 0))
            path_stats["dsa_selection_spill_bytes_written"] = int(
                dsa_state.stats.get("selection_spill_bytes_written", 0))
            path_stats["dsa_selection_spill_bytes_read"] = int(
                dsa_state.stats.get("selection_spill_bytes_read", 0))
            path_stats["dsa_selection_spill_reads"] = int(
                dsa_state.stats.get("selection_spill_reads", 0))
            path_stats["dsa_selection_spill_flushes"] = int(
                dsa_state.stats.get("selection_spill_flushes", 0))
            path_stats["dsa_selection_spill_write_s"] = float(
                dsa_state.stats.get("selection_spill_write_s", 0.0))
            path_stats["dsa_selection_spill_read_s"] = float(
                dsa_state.stats.get("selection_spill_read_s", 0.0))
            path_stats["dsa_selection_score_s"] = float(
                dsa_state.stats.get("selection_score_s", 0.0))
            path_stats["dsa_preselection_s"] = float(
                dsa_state.stats.get("preselection_s", 0.0))
            path_stats["dsa_index_observe_s"] = float(
                dsa_state.stats.get("index_observe_s", 0.0))
        if self.cfg.model_type == "glm_moe_dsa":
            glm53_memory = getattr(
                self, "_glm53_layer_stationary_stats", {}) or {}
            path_stats["glm53_layer_stationary_host_spool"] = int(
                self.rc.glm53_layer_stationary_host_spool)
            path_stats["glm53_layer_stationary_memory_samples"] = int(
                glm53_memory.get("memory_samples", 0))
            path_stats["glm53_layer_stationary_peak_metal_bytes"] = int(
                glm53_memory.get("peak_metal_bytes", 0))
            for phase in (
                    "initial_carrier", "attention", "ffn_hc_pre", "mlp",
                    "ffn_hc_post"):
                path_stats[
                    f"glm53_layer_stationary_{phase}_active_peak_bytes"
                ] = int(glm53_memory.get(
                    f"{phase}_active_peak_bytes", 0))
            path_stats["glm53_layer_stationary_tile_width"] = int(
                glm53_memory.get("tile_width", 0))
            path_stats["glm53_layer_stationary_positions"] = int(
                glm53_memory.get("positions", 0))
            path_stats["glm53_layer_stationary_sweep_positions"] = int(
                glm53_memory.get("sweep_positions", 0))
            path_stats["glm53_layer_stationary_sweeps"] = int(
                glm53_memory.get("sweeps", 0))
            for metric in (
                    "transient_reservation_calls",
                    "transient_reservation_bytes",
                    "transient_reservation_margin_bytes",
                    "transient_reservation_first_margin_calls",
                    "transient_reservation_recurring_calls"):
                path_stats[f"glm53_layer_stationary_{metric}"] = int(
                    glm53_memory.get(metric, 0))
            path_stats[
                "glm53_layer_stationary_transient_reservation_s"
            ] = float(glm53_memory.get(
                "transient_reservation_s", 0.0))
            for metric in (
                    "host_spool_h2d_bytes", "host_spool_d2h_bytes",
                    "host_spool_peak_host_bytes"):
                path_stats[f"glm53_layer_stationary_{metric}"] = int(
                    glm53_memory.get(metric, 0))
            path_stats[
                "glm53_layer_stationary_host_spool_copy_s"
            ] = float(glm53_memory.get("host_spool_copy_s", 0.0))
        if self.cfg.model_type == "glm5_next":
            glm53_memory = getattr(
                self, "_glm53_layer_stationary_stats", {}) or {}
            path_stats["glm53_sparse_absorbed_mla"] = int(
                self.rc.glm53_sparse_absorbed_mla)
            path_stats["glm53_sparse_fused_attention"] = int(
                self.rc.glm53_sparse_fused_attention)
            path_stats["glm53_sparse_fused_kv_int8"] = int(
                self.rc.glm53_sparse_fused_kv_int8)
            path_stats["glm53_coalesced_expert_positions"] = int(
                self.rc.glm53_coalesced_expert_positions)
            path_stats["glm53_coalesced_expert_position_limit"] = int(
                self.rc.glm53_coalesced_expert_max_positions)
            path_stats["glm53_coalesced_expert_gemm_calls"] = int(
                glm53_memory.get("coalesced_expert_gemm_calls", 0))
            path_stats["glm53_coalesced_expert_max_positions"] = int(
                glm53_memory.get("coalesced_expert_max_positions", 0))
            path_stats["glm53_coalesced_expert_split_experts"] = int(
                glm53_memory.get("coalesced_expert_split_experts", 0))
            for metric in (
                    "layers", "input_positions", "route_assignments",
                    "unique_experts", "max_unique_experts",
                    "max_expert_routes", "gemm_input_positions",
                    "gemm_full_chunks"):
                path_stats[f"glm53_coalesced_expert_{metric}"] = int(
                    glm53_memory.get(f"coalesced_expert_{metric}", 0))
            for metric in (
                    "layers", "tiles", "swiglu_calls", "rows", "max_rows",
                    "rows_1_calls", "rows_2_calls", "rows_3_4_calls",
                    "rows_5_8_calls", "rows_9_16_calls",
                    "rows_17_32_calls", "rows_33_plus_calls"):
                path_stats[f"glm53_exact_expert_{metric}"] = int(
                    glm53_memory.get(f"exact_expert_{metric}", 0))
            path_stats["glm53_incremental_dsa_pool"] = int(
                self.rc.glm53_incremental_dsa_pool)
            path_stats["glm53_compiled_kda_prefill"] = int(
                self.rc.glm53_compiled_kda_prefill)
            path_stats["glm53_compiled_kda_segment"] = int(
                self.rc.glm53_compiled_kda_segment)
            path_stats["glm53_native_fused_kda_prefill"] = int(
                self.rc.glm53_native_fused_kda_prefill)
            path_stats["glm53_layer_stationary_host_spool"] = int(
                self.rc.glm53_layer_stationary_host_spool)
            path_stats["glm53_layer_stationary_memory_samples"] = int(
                glm53_memory.get("memory_samples", 0))
            path_stats["glm53_layer_stationary_peak_metal_bytes"] = int(
                glm53_memory.get("peak_metal_bytes", 0))
            for phase in (
                    "initial_carrier", "attention", "ffn_hc_pre", "mlp",
                    "ffn_hc_post"):
                path_stats[
                    f"glm53_layer_stationary_{phase}_active_peak_bytes"
                ] = int(glm53_memory.get(
                    f"{phase}_active_peak_bytes", 0))
            path_stats["glm53_layer_stationary_tile_width"] = int(
                glm53_memory.get("tile_width", 0))
            path_stats["glm53_layer_stationary_positions"] = int(
                glm53_memory.get("positions", 0))
            path_stats["glm53_layer_stationary_sweep_positions"] = int(
                glm53_memory.get("sweep_positions", 0))
            path_stats["glm53_layer_stationary_sweeps"] = int(
                glm53_memory.get("sweeps", 0))
            for metric in (
                    "host_spool_h2d_bytes", "host_spool_d2h_bytes",
                    "host_spool_peak_host_bytes"):
                path_stats[f"glm53_layer_stationary_{metric}"] = int(
                    glm53_memory.get(metric, 0))
            path_stats["glm53_layer_stationary_host_spool_copy_s"] = float(
                glm53_memory.get("host_spool_copy_s", 0.0))
            path_stats["glm53_sparse_fused_calls"] = int(
                glm53_memory.get("sparse_fused_calls", 0))
            path_stats["glm53_sparse_fused_positions"] = int(
                glm53_memory.get("sparse_fused_positions", 0))
            path_stats["glm53_sparse_fused_selected_rows"] = int(
                glm53_memory.get("sparse_fused_selected_rows", 0))
            path_stats["glm53_layer_stationary_weight_wait_s"] = float(
                glm53_memory.get("weight_wait_s", 0.0))
            path_stats["glm53_layer_stationary_attention_s"] = float(
                glm53_memory.get("attention_residual_s", 0.0))
            path_stats["glm53_layer_stationary_kda_attention_s"] = float(
                glm53_memory.get("kda_attention_s", 0.0))
            path_stats["glm53_layer_stationary_mla_attention_s"] = float(
                glm53_memory.get("mla_attention_s", 0.0))
            path_stats["glm53_layer_stationary_ffn_hc_pre_s"] = float(
                glm53_memory.get("ffn_hc_pre_s", 0.0))
            path_stats["glm53_layer_stationary_mlp_s"] = float(
                glm53_memory.get("mlp_s", 0.0))
            path_stats["glm53_layer_stationary_ffn_hc_post_s"] = float(
                glm53_memory.get("ffn_hc_post_s", 0.0))
            if dsa_state is not None:
                path_stats["glm53_dsa_pool_rows_computed"] = int(
                    dsa_state.stats.get("pool_rows_computed", 0))
                path_stats["glm53_dsa_pool_rows_reused"] = int(
                    dsa_state.stats.get("pool_rows_reused", 0))
                path_stats["glm53_dsa_pool_build_s"] = float(
                    dsa_state.stats.get("pool_build_s", 0.0))
                path_stats["glm53_dsa_selection_s"] = float(
                    dsa_state.stats.get("selection_s", 0.0))
                path_stats["glm53_dsa_packed_capacity_grows"] = int(
                    dsa_state.stats.get("packed_capacity_grows", 0))
                path_stats["glm53_dsa_packed_rows_copied"] = int(
                    dsa_state.stats.get("packed_rows_copied", 0))
                path_stats["glm53_dsa_packed_rows_appended"] = int(
                    dsa_state.stats.get("packed_rows_appended", 0))
                path_stats["glm53_dsa_packed_capacity_rows_peak"] = int(
                    dsa_state.stats.get("packed_capacity_rows_peak", 0))
                path_stats["glm53_dsa_pool_capacity_grows"] = int(
                    dsa_state.stats.get("pool_capacity_grows", 0))
                path_stats["glm53_dsa_pool_rows_copied"] = int(
                    dsa_state.stats.get("pool_rows_copied", 0))
                path_stats["glm53_dsa_pool_capacity_rows_peak"] = int(
                    dsa_state.stats.get("pool_capacity_rows_peak", 0))
                path_stats["glm53_dsa_pool_metadata_rows_avoided"] = int(
                    dsa_state.stats.get("pool_metadata_rows_avoided", 0))
        path_stats["expert_compute_batches"] = self._expert_compute_batches
        path_stats["max_experts_per_compute_batch"] = self._max_experts_per_compute_batch
        path_stats["adaptive_expert_batch_clamps"] = self._adaptive_expert_batch_clamps
        path_stats["min_adaptive_expert_batch"] = self._min_adaptive_expert_batch
        path_stats["expert_batch_prefetch"] = int(
            self._expert_batch_executor is not None)
        path_stats["expert_batch_prefetch_prefill_only"] = int(
            self.rc.qwen4_expert_batch_prefetch_prefill_only)
        path_stats["expert_batch_prefetch_submitted"] = (
            self._expert_batch_prefetch_submitted)
        path_stats["expert_batch_prefetch_wait_s"] = (
            self._expert_batch_prefetch_wait_s)
        path_stats["expert_batch_prefetch_hidden_s"] = (
            self._expert_batch_prefetch_hidden_s)
        path_stats["expert_batch_prefetch_depth"] = int(getattr(
            self.rc, "expert_batch_prefetch_depth", 1))
        path_stats["expert_batch_prefetch_workers"] = int(getattr(
            self.rc, "expert_batch_prefetch_workers", 1))
        path_stats["expert_batch_prefetch_max_futures"] = (
            self._expert_batch_prefetch_max_futures)
        for phase in ("prefill", "decode"):
            path_stats[f"expert_batch_prefetch_{phase}_submitted"] = (
                self._expert_batch_prefetch_submitted_by_phase[phase])
            path_stats[f"expert_batch_prefetch_{phase}_wait_s"] = (
                self._expert_batch_prefetch_wait_s_by_phase[phase])
            path_stats[f"expert_batch_prefetch_{phase}_hidden_s"] = (
                self._expert_batch_prefetch_hidden_s_by_phase[phase])
        path_stats["expert_shared_overlap_layers"] = (
            self._expert_shared_overlap_layers)
        path_stats["qwen4_serial_verify_union_layers"] = (
            self._qwen4_serial_verify_union_layers)
        path_stats["qwen4_serial_verify_expert_slots"] = (
            self._qwen4_serial_verify_expert_slots)
        path_stats["qwen4_serial_verify_union_experts"] = (
            self._qwen4_serial_verify_union_experts)
        path_stats["qwen4_serial_verify_expert_pages_avoided"] = (
            self._qwen4_serial_verify_expert_pages_avoided)
        path_stats["qwen4_serial_verify_union_fetch_s"] = (
            self._qwen4_serial_verify_union_fetch_s)
        path_stats["qwen4_serial_verify_page_prepare_s"] = (
            self._qwen4_serial_verify_page_prepare_s)
        path_stats["qwen4_serial_verify_weight_wait_s"] = (
            self._qwen4_serial_verify_weight_wait_s)
        path_stats["qwen4_serial_verify_reserve_s"] = (
            self._qwen4_serial_verify_reserve_s)
        path_stats["qwen4_serial_verify_linear_compute_s"] = (
            self._qwen4_serial_verify_linear_compute_s)
        path_stats["qwen4_serial_verify_full_compute_s"] = (
            self._qwen4_serial_verify_full_compute_s)
        path_stats["qwen4_serial_verify_linear_layers"] = (
            self._qwen4_serial_verify_linear_layers)
        path_stats["qwen4_serial_verify_full_layers"] = (
            self._qwen4_serial_verify_full_layers)
        path_stats["qwen4_serial_verify_head_s"] = (
            self._qwen4_serial_verify_head_s)
        path_stats["qwen4_serial_verify_pipelined_expert_layers"] = (
            self._qwen4_serial_verify_pipelined_expert_layers)
        path_stats["qwen4_serial_verify_exact_bf16_gemv"] = int(
            self.rc.qwen4_serial_verify_exact_bf16_gemv)
        path_stats["qwen4_serial_verify_exact_bf16_calls"] = int(
            self._qwen4_serial_verify_exact_bf16_calls)
        path_stats["qwen4_serial_verify_exact_bf16_rows"] = int(
            self._qwen4_serial_verify_exact_bf16_rows)
        path_stats["qwen4_serial_verify_exact_bf16_fallback_calls"] = int(
            self._qwen4_serial_verify_exact_bf16_fallback_calls)
        for reason, count in (
            self._qwen4_serial_verify_exact_bf16_fallback_reasons.items()
        ):
            path_stats[
                f"qwen4_serial_verify_exact_bf16_fallback_{reason}_calls"
            ] = int(count)
        qwen4_state = getattr(kv, "qwen4_cache", None)
        if qwen4_state is not None:
            path_stats.update(qwen4_state.qsa_pool_cache_stats())
        overlap = self._expert_route_overlap_totals
        for key in (
            "calls",
            "positions",
            "selected_slots",
            "union_experts",
            "adjacent_pairs",
            "within_call_pairs",
            "cross_call_pairs",
            "adjacent_intersection_experts",
            "adjacent_union_experts",
            "adjacent_current_experts",
            "exact_adjacent_pairs",
            "cross_call_intersection_experts",
            "cross_call_current_experts",
        ):
            path_stats[f"expert_route_{key}"] = int(overlap.get(key, 0))
        path_stats["expert_route_adjacent_reuse_fraction"] = (
            overlap.get("adjacent_intersection_experts", 0)
            / max(1, overlap.get("adjacent_current_experts", 0))
        )
        path_stats["expert_route_adjacent_jaccard"] = (
            overlap.get("adjacent_intersection_experts", 0)
            / max(1, overlap.get("adjacent_union_experts", 0))
        )
        path_stats["expert_route_union_efficiency"] = (
            overlap.get("selected_slots", 0)
            / max(1, overlap.get("union_experts", 0))
        )
        path_stats["expert_route_cross_call_reuse_fraction"] = (
            overlap.get("cross_call_intersection_experts", 0)
            / max(1, overlap.get("cross_call_current_experts", 0))
        )
        path_stats["expert_route_cross_call_reusable_storage_bytes_upper_bound"] = (
            int(overlap.get("cross_call_intersection_experts", 0))
            * int(self._expert_storage_page_bytes)
        )
        path_stats["expert_resident_page_bytes_estimate"] = self._expert_page_bytes
        path_stats["expert_storage_page_bytes_estimate"] = (
            self._expert_storage_page_bytes)
        path_stats["expert_fetch_page_bytes_estimate"] = self._expert_fetch_page_bytes
        path_stats["configured_expert_fetch_batch"] = self.rc.expert_fetch_batch
        path_stats["configured_expert_compute_batch"] = (
            self.rc.expert_compute_batch)
        path_stats["auto_compact_expert_fetch_batch"] = (
            self._auto_compact_expert_batch)
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
        path_stats["paged_kv_page_native_calls"] = int(
            getattr(paged_stats, "page_native_calls", 0) or 0)
        path_stats["paged_kv_page_native_groups"] = int(
            getattr(paged_stats, "page_native_groups", 0) or 0)
        path_stats["paged_kv_page_native_positions"] = int(
            getattr(paged_stats, "page_native_positions", 0) or 0)
        path_stats["paged_kv_page_native_seconds"] = float(
            getattr(paged_stats, "page_native_s", 0.0) or 0.0)
        if getattr(kv, "position_free", False):
            # The view exists only to make this request's long decode use MLX's
            # fast pre-rotated SDPA. The retained hot slot owns shared physical
            # pages only, so subsequent edited branches do not duplicate KV.
            kv.drop_rotated_view()

        if (hot_eligible and (
                isinstance(kv, KVCache) or self.rc.paged_kv_persist)
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
            qwen4_state = getattr(kv, "qwen4_cache", None)
            if qwen4_state is not None:
                qwen4_state.synchronize()
            full_tokens = tuple(tokens + generated[:-1])
            segment_chain: tuple[str, ...] = ()
            if self._hot_kv_persist is not None:
                persist_t0 = time.perf_counter()
                endpoint_parent_chain = persist_parent_chain
                endpoint_parent_covered = persist_parent_covered
                # Qwen's useful next-turn endpoint is the exact chat-template
                # boundary fork captured during prefill, not the later
                # generation-scaffold endpoint. Persist that hybrid state as a
                # typed, state-only checkpoint before the ordinary endpoint.
                # Its full-attention KV segments also become the immutable
                # parent of the later full endpoint, so this adds only the
                # recurrent checkpoint payload rather than duplicating prefix
                # KV bytes.
                stable_boundary_available = bool(
                    self.cfg.model_type in (
                        "qwen3_5", "qwen3_5_moe", "qwen4_exp")
                    and boundary_fork_kv is not None
                    and 0 < boundary_fork_tokens <= len(tokens)
                )
                stable_boundary_persistable = bool(
                    stable_boundary_available
                    and _stable_boundary_persistence_allowed(
                        self._hot_kv_persist,
                        approximate=prompt_state_approximate,
                    )
                )
                if stable_boundary_persistable:
                    boundary_recurrent = getattr(
                        boundary_fork_kv, "kda_cache", None)
                    if boundary_recurrent is None:
                        raise RuntimeError(
                            "Qwen stable boundary is missing recurrent state")
                    boundary_recurrent.synchronize()
                    boundary_qwen4 = getattr(
                        boundary_fork_kv, "qwen4_cache", None)
                    if self.cfg.model_type == "qwen4_exp":
                        if boundary_qwen4 is None:
                            raise RuntimeError(
                                "Qwen4 stable boundary is missing QSA/PLE state")
                        boundary_qwen4.synchronize()
                    boundary_segment_chain = self._hot_kv_persist.save(
                        parent_chain=persist_parent_chain,
                        parent_covered=persist_parent_covered,
                        tokens=tuple(tokens[:boundary_fork_tokens]),
                        kv=boundary_fork_kv,
                        logits=None,
                        prompt_logits=None,
                        prompt_length=boundary_fork_tokens,
                        reusable_prefix=boundary_fork_tokens,
                        approximate=prompt_state_approximate,
                        tool_capsules=(),
                        cache_namespace=cache_namespace,
                        checkpoint_kind="stable_prefix",
                    )
                    endpoint_parent_chain = boundary_segment_chain
                    endpoint_parent_covered = boundary_fork_tokens
                    path_stats[
                        "hot_prompt_hybrid_prefix_snapshot_tokens"] = (
                            boundary_fork_tokens)
                elif stable_boundary_available:
                    # The configured mixed-depth schedule is content-blind,
                    # but a short request may fit wholly inside its suffix
                    # window and therefore remain exact.  Such state is a
                    # valid in-memory boundary fork, not a mixed-depth disk
                    # snapshot.  Skipping it is both safe and observable.
                    path_stats[
                        "hot_prompt_mixed_depth_snapshot_skipped_exact"] = 1
                segment_chain = self._hot_kv_persist.save(
                    parent_chain=endpoint_parent_chain,
                    parent_covered=endpoint_parent_covered,
                    tokens=full_tokens,
                    kv=kv,
                    logits=logits,
                    prompt_logits=prompt_endpoint_logits,
                    exact_hidden=(
                        self._h_last
                        if self.cfg.model_type == "qwen4_exp" else None),
                    prompt_length=len(tokens),
                    reusable_prefix=reusable_watermark,
                    approximate=prompt_state_approximate,
                    tool_capsules=tuple(getattr(prompt, "tool_capsules", ())),
                    cache_namespace=cache_namespace,
                )
                path_stats["hot_prompt_kv_persist_write_s"] = (
                    time.perf_counter() - persist_t0)
                respilled = self._respill_completed_k3_state(kv)
                path_stats["hot_prompt_k3_respill_kda_layers"] = respilled[
                    "kda_layers"]
                path_stats["hot_prompt_k3_respill_mla_layers"] = respilled[
                    "mla_layers"]
            if isinstance(kv, KVCache):
                new_slot = self._new_hot_prompt_slot(
                    recurrent_exact_only=recurrent_exact_only,
                    boundary_fork_kv=boundary_fork_kv,
                    boundary_fork_tokens=boundary_fork_tokens,
                    tokens=tokens, full_tokens=full_tokens, kv=kv, logits=logits,
                    prompt_endpoint_logits=prompt_endpoint_logits,
                    reusable_watermark=reusable_watermark,
                    prompt_state_approximate=prompt_state_approximate,
                    tool_capsules=tuple(getattr(prompt, "tool_capsules", ())),
                    segment_chain=segment_chain,
                    boundary_segment_chain=boundary_segment_chain,
                    cache_namespace=cache_namespace,
                )
                capacity_count, capacity_bytes = self._append_hot_prompt_slot(
                    new_slot)
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
        # A configured serving floor is also a post-response retention
        # invariant. Admission protects it before large allocations, but a
        # cold MoE sweep can finish with a full weight cache plus the newly
        # retained prompt endpoint and land just below the floor. Shed only the
        # measured deficit (plus one small sampling pad) from consumed LRU
        # weight pages after all model arithmetic and endpoint synchronization.
        # ``trim_to`` leaves the configured admission budget unchanged, so the
        # next request can refill normally instead of running the entire model
        # under a smaller cache and changing its latency/trajectory.
        postgen_floor = int(
            self.rc.qwen_postgen_min_available_mb * 1_000_000)
        postgen_available_before = int(psutil.virtual_memory().available)
        if postgen_floor > 0 and postgen_available_before < postgen_floor:
            cache_before_trim = int(self.cache.total_bytes)
            requested_reclaim = (
                postgen_floor - postgen_available_before + 128_000_000)
            released = self.cache.trim_to(
                cache_before_trim - requested_reclaim)
            path_stats["postgen_weight_cache_trim"] = 1
            path_stats["postgen_weight_cache_trim_requested_bytes"] = int(
                requested_reclaim)
            path_stats["postgen_weight_cache_trim_released_bytes"] = int(
                released)
            path_stats["postgen_weight_cache_trim_available_before_bytes"] = (
                postgen_available_before)
            path_stats["postgen_weight_cache_trim_available_after_bytes"] = int(
                psutil.virtual_memory().available)
        request_cache_after = _cache_io_snapshot(self)
        kda_cache = getattr(kv, "kda_cache", None)
        if kda_cache is not None and getattr(
            kda_cache, "spill_enabled", False
        ):
            self._last_k3_kda_spill_stats = kda_cache.spill_stats()
            path_stats["k3_kda_spill"] = dict(
                self._last_k3_kda_spill_stats)
        if getattr(kv, "latent_spill_enabled", False):
            spill_stats = kv.latent_spill_stats()
            if self.cfg.model_type == "glm_moe_dsa":
                self._last_glm_dsa_mla_kv_spill_stats = spill_stats
                path_stats["glm_dsa_mla_kv_spill"] = dict(spill_stats)
                for key, value in spill_stats.items():
                    path_stats[f"glm_dsa_mla_kv_spill_{key}"] = int(value)
            else:
                self._last_k3_mla_kv_spill_stats = spill_stats
                path_stats["k3_mla_kv_spill"] = dict(spill_stats)
        _record_cache_io_delta(
            self, request_cache_before, path_stats, after=request_cache_after)
        _record_cache_io_delta(
            self, request_cache_before, path_stats, prefix="prefill_",
            after=prefill_cache_after)
        _record_cache_io_delta(
            self, prefill_cache_after, path_stats, prefix="decode_",
            after=request_cache_after)
        if qwen4_expert_before is not None:
            qwen4_expert_after = self.store.qwen4_fused_expert_snapshot()
            for key in ("calls", "extents", "requested_tensors", "bytes"):
                path_stats[f"qwen4_fused_expert_{key}"] = max(
                    0, int(qwen4_expert_after[key])
                    - int(qwen4_expert_before[key]))
            path_stats["qwen4_fused_expert_virtual_tensors"] = int(
                qwen4_expert_after["virtual_tensors"])
        _record_direct_io_delta(self, direct_io_before, path_stats)
        if qwen4_ple_before is not None:
            qwen4_ple_after = self._qwen4_ple_rows.telemetry()
            for key in (
                "read_calls", "read_extents", "rows_requested",
                "unique_rows_read", "bytes_read", "cache_hits",
            ):
                path_stats[f"qwen4_ple_{key}"] = max(
                    0, int(qwen4_ple_after[key])
                    - int(qwen4_ple_before[key]))
            path_stats["qwen4_ple_source_fingerprint"] = (
                qwen4_ple_after["source_fingerprint"])
            path_stats["qwen4_ple_source_revision"] = (
                qwen4_ple_after["source_revision"])
            path_stats["qwen4_ple_source_verified_release_hash"] = int(
                qwen4_ple_after["source_verified_release_hash"])
            path_stats["qwen4_ple_storage_dtype"] = str(
                qwen4_ple_after["storage_dtype"])
            for key in (
                "storage_row_bytes", "output_row_bytes", "scale_bytes_read",
            ):
                path_stats[f"qwen4_ple_{key}"] = int(qwen4_ple_after[key])
            for key, value in self._qwen4_host_spool_stats.items():
                path_stats[f"qwen4_host_spool_{key}"] = value
            path_stats["qwen4_phase_lm_head"] = int(
                self.rc.qwen4_phase_lm_head)
            path_stats["qwen4_phase_lm_head_bytes"] = int(
                self._qwen4_phase_head_bytes)
            path_stats["qwen4_phase_lm_head_suspend_calls"] = int(
                self._qwen4_phase_head_suspend_calls)
            path_stats["qwen4_phase_lm_head_suspend_bytes"] = int(
                self._qwen4_phase_head_suspend_bytes)
            path_stats["qwen4_phase_lm_head_suspend_s"] = float(
                self._qwen4_phase_head_suspend_s)
            path_stats["qwen4_phase_lm_head_restore_calls"] = int(
                self._qwen4_phase_head_restore_calls)
            path_stats["qwen4_phase_lm_head_restore_successes"] = int(
                self._qwen4_phase_head_restore_successes)
            path_stats["qwen4_phase_lm_head_restore_refusals"] = int(
                self._qwen4_phase_head_restore_refusals)
            path_stats["qwen4_phase_lm_head_restore_s"] = float(
                self._qwen4_phase_head_restore_s)
            path_stats["qwen4_serial_verify_suspend_lm_head"] = int(
                self.rc.qwen4_serial_verify_suspend_lm_head)
            path_stats["qwen4_serial_verify_head_suspend_calls"] = int(
                self._qwen4_serial_verify_head_suspend_calls)
            path_stats["qwen4_serial_verify_head_suspend_bytes"] = int(
                self._qwen4_serial_verify_head_suspend_bytes)
            path_stats[
                "qwen4_serial_verify_head_restore_trim_bytes"
            ] = int(self._qwen4_serial_verify_head_restore_trim_bytes)
        if reranked_telemetry_before:
            reranked_after = reranked_head.telemetry_snapshot()
            for key, value in reranked_after.items():
                path_stats[f"reranked_lm_head_{key}"] = max(
                    0, int(value) - int(reranked_telemetry_before.get(key, 0)))
            recall_probes = path_stats[
                "reranked_lm_head_candidate_recall_probes"]
            recall_hits = path_stats[
                "reranked_lm_head_candidate_recall_hits"]
            path_stats["reranked_lm_head_candidate_recall"] = (
                recall_hits / recall_probes if recall_probes else None)
        total_s = time.perf_counter() - request_t0
        execution_profile = (
            self._request_profiler.result(total_s)
            if self._request_profiler is not None else None)
        result = {
            "text": final_text,
            "tokens": generated,
            "prefill_s": prefill_s,
            "decode_s": sum(tok_times),
            "first_token_s": first_token_s,
            "total_s": total_s,
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
        if execution_profile is not None:
            result["execution_profile"] = execution_profile
        retain_internal_paged_kv = bool(
            getattr(prompt, "retain_paged_kv_after_generate", False))
        if (((self.rc.release_paged_kv_after_generate and self.rc.max_kv_mb)
                or force_adaptive_paged)
                and not retain_internal_paged_kv):
            self.last_kv = None
            self._release_kv(kv)
            mx.clear_cache()
        self._completed_generations += 1
        return result

    def generate_with_memory_retry(
            self, prompt: str, max_tokens: int = 64, on_token=None, stop=None,
            on_progress=None, sampling: SamplingParams | None = None,
            constraint=None) -> dict:
        """Retry an unstarted hybrid prefill on progressively smaller chunks.

        Qwen3.5/3.6's recurrent state cannot use the ordinary disk-paged KV
        fallback, while large quantized dense Qwen sweeps can outgrow the
        headroom observed at request start.  When
        the governor refuses an allocation *before the first token is sampled*,
        discard the partial recurrent/KV state and replay from the original
        prompt at the next rung: 512 -> 128 -> 32 -> 8 -> 1. GLM's explicit
        coalesced-expert candidate first reduces its gathered-position ceiling
        because changing the underlying tile width does not bound an expert
        gathered across all tiles. This preserves the configured arithmetic
        class and favors arbitrarily slow bounded prefill over a hard error.
        No retry is allowed after sampling begins.
        """
        failed_seconds = 0.0
        retry_chunks: list[int] = []
        retry_coalesced_limits: list[int] = []
        retry_cleanup: list[dict[str, int]] = []
        retry_failures: list[str] = []
        self._hybrid_retry_chunk_ceiling = 0
        try:
            while True:
                attempt_t0 = time.perf_counter()
                try:
                    result = self.generate(
                        prompt, max_tokens, on_token=on_token, stop=stop,
                        on_progress=on_progress, sampling=sampling,
                        constraint=constraint)
                    if retry_chunks or retry_coalesced_limits:
                        result["prefill_s"] += failed_seconds
                        result["first_token_s"] += failed_seconds
                        result["total_s"] += failed_seconds
                        stats = result.setdefault("path_stats", {})
                        stats["memory_prefill_retries"] = (
                            len(retry_chunks) + len(retry_coalesced_limits))
                        stats["memory_prefill_retry_chunks"] = retry_chunks
                        stats["memory_prefill_retry_coalesced_limits"] = (
                            retry_coalesced_limits)
                        stats["memory_prefill_retry_seconds"] = failed_seconds
                        stats["memory_prefill_retry_cleanup"] = retry_cleanup
                        stats["memory_prefill_retry_failures"] = retry_failures
                    return result
                except MemoryError as error:
                    failed_seconds += time.perf_counter() - attempt_t0
                    if (getattr(self, "_generation_sampled_tokens", 0)
                            or not self._memory_prefill_retry_applies()):
                        raise
                    current = max(1, int(self.rc.prefill_chunk_size or 1))
                    current_coalesced_limit = int(getattr(
                        self.rc, "glm53_coalesced_expert_max_positions",
                        512) or 512)
                    next_coalesced_limit = 0
                    if (self.cfg.model_type == "glm5_next"
                            and bool(getattr(
                                self.rc,
                                "glm53_coalesced_expert_positions", False))
                            and current_coalesced_limit > 128):
                        next_coalesced_limit = next(
                            (candidate for candidate in (
                                2048, 1024, 512, 256, 128)
                             if candidate < current_coalesced_limit),
                            0,
                        )
                    if self.cfg.model_type == "glm5_next":
                        retry_ladder = (128, 64, 32, 8, 1)
                    elif self.cfg.model_type == "glm_moe_dsa":
                        retry_ladder = (8, 4, 2, 1)
                    else:
                        retry_ladder = (128, 32, 8, 1)
                    next_chunk = 0
                    if not next_coalesced_limit:
                        next_chunk = next(
                            (candidate for candidate in retry_ladder
                             if candidate < current),
                            0,
                        )
                        if not next_chunk:
                            raise
                        retry_chunks.append(next_chunk)
                    else:
                        retry_coalesced_limits.append(
                            next_coalesced_limit)
                    retry_failures.append(str(error))
                    if on_progress is not None:
                        # Surface a privacy-safe retry boundary before the
                        # expensive replay begins. Long captured-request gates
                        # can fail fast on this event instead of silently
                        # spending hours on a known-bad smaller tile. No prompt,
                        # route, tensor, or exception text leaves the engine.
                        retry_count = (
                            len(retry_chunks)
                            + len(retry_coalesced_limits)
                        )
                        retry_total = len(retry_ladder) + (
                            5 if (
                                self.cfg.model_type == "glm5_next"
                                and bool(getattr(
                                    self.rc,
                                    "glm53_coalesced_expert_positions",
                                    False))
                            ) else 0
                        )
                        retry_progress = {
                            "phase": "memory_retry",
                            "completed_retries": retry_count,
                            "total_retries": retry_total,
                            "retry_chunk": next_chunk,
                            "retry_coalesced_expert_max_positions": (
                                next_coalesced_limit),
                        }
                        retry_progress.update(
                            _memory_retry_diagnostic(error))
                        on_progress(retry_progress)
                    self.discard_failed_request_state()
                    if next_coalesced_limit:
                        self.rc.glm53_coalesced_expert_max_positions = (
                            next_coalesced_limit)
                    else:
                        self._hybrid_retry_chunk_ceiling = next_chunk
                        self.rc.prefill_chunk_size = next_chunk
                        self.rc.hot_prompt_kv_chunk_size = next_chunk
                    if self.rc.adaptive_chunk_size and next_chunk:
                        # F68's AdaptiveChunkController only bounds the
                        # sweep's own compute-scratch peak -- it has no
                        # visibility into _fetch_experts' separate
                        # governor.reserve() call, which can refuse even a
                        # chunk the controller judged safe (more distinct
                        # tokens in a bigger chunk route to more distinct
                        # experts, inflating that unrelated one-shot
                        # reservation). A MemoryError reaching here means
                        # growth already outran that other budget once, so
                        # pin a hard, non-adaptive ceiling instead of
                        # handing a fresh controller a smaller seed it will
                        # just grow back out of -- consistent with this
                        # file's existing "never auto-restore after a
                        # shrink" stance elsewhere (see adaptive_chunk.py).
                        #
                        # A dedicated flag, NOT self.rc.adaptive_chunk_size
                        # itself: live-confirmed 2026-07-29 that flipping
                        # the config field directly breaks retry eligibility
                        # for any SUBSEQUENT failure in this same loop, since
                        # _memory_prefill_retry_applies() (below) reads that
                        # same field -- a real MemoryError on the pinned
                        # retry attempt silently lost retry coverage instead
                        # of continuing down the chunk ladder. This flag is
                        # request-independent (survives for the engine's
                        # lifetime, matching prefill_chunk_size's own
                        # never-restored reduction above) but orthogonal to
                        # the config field eligibility depends on.
                        self._adaptive_chunk_pinned_after_retry = True
                    print(
                        "[prefill] memory refusal before first sample: "
                        f"{error}; retrying from scratch at " + (
                            f"coalesced_expert_max_positions="
                            f"{next_coalesced_limit}"
                            if next_coalesced_limit
                            else f"chunk={next_chunk}"),
                        flush=True,
                    )
                # Run allocator cleanup only after leaving the ``except``
                # suite. While an exception is being handled, its traceback
                # still owns the failed generate() frame and every carrier
                # tensor in that frame. Clearing MLX inside the handler cannot
                # release those live references, so the next attempt inherits
                # both the old allocations and their historical peak.
                active_before = int(mx.get_active_memory())
                failed_peak = int(mx.get_peak_memory())
                mx.clear_cache()
                active_after = int(mx.get_active_memory())
                # ``_note_true_peak`` recorded the process-wide peak before
                # the MemoryError. Reset only MLX's per-attempt high-water
                # mark so the next attempt's hard guard is not poisoned by a
                # failed 64/32-token tile.
                mx.reset_peak_memory()
                retry_cleanup.append({
                    "chunk": int(self.rc.prefill_chunk_size),
                    "coalesced_expert_max_positions": int(getattr(
                        self.rc, "glm53_coalesced_expert_max_positions",
                        0) or 0),
                    "active_before_bytes": active_before,
                    "active_after_bytes": active_after,
                    "released_bytes": max(0, active_before - active_after),
                    "failed_peak_bytes": failed_peak,
                })
        finally:
            self._hybrid_retry_chunk_ceiling = 0

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
            f"resident_fast: decode_sweeps={self._resident_fast_decode_sweeps} "
            f"prefill_sweeps={self._resident_fast_prefill_sweeps} "
            f"moe_sweeps={self._resident_moe_sweeps} "
            f"disabled_for_request={self._disable_resident_fast_for_request}",
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
        self._glm53_vision_prompt_cache = None
        vision_embeddings = getattr(self, "_vision_embedding_cache", None)
        if vision_embeddings is not None:
            vision_embeddings.clear()
        glm53_embeddings = getattr(self, "_glm53_vision_embedding_cache", None)
        if glm53_embeddings is not None:
            glm53_embeddings.clear()

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

    def _hybrid_chunk_size_applies(self) -> bool:
        """F95: per-conversation prefill_chunk_size adaptivity is scoped to
        qwen3_5/qwen3_5_moe hot_prompt_kv targets with durable persistence
        OFF -- persistence bakes one engine-wide chunk size into its
        on-disk format (HotPromptKVPersistence), incompatible with varying
        it per conversation until a follow-up extends that format too."""
        return (self._hot_kv_persist is None
                and self.cfg.model_type in ("qwen3_5", "qwen3_5_moe"))

    def _memory_prefill_retry_applies(self) -> bool:
        """Whether a fresh, unsampled prefill can be replayed more slowly."""
        if self._hybrid_chunk_size_applies():
            return True
        if (
            self._hot_kv_persist is None
            and self.cfg.model_type == "kimi_k3"
            and self.rc.kimi_k3_prefill_tile_policy == "prompt-length"
        ):
            return True
        if (
            self._hot_kv_persist is None
            and self.cfg.model_type == "glm5_next"
            and self.rc.layer_stationary_prefill
        ):
            # A real 46,849-token GLM-5.3 harness prompt reached 8.518GB
            # during a 128-position attention tile and correctly tripped the
            # hard 8.5GB cap before sampling. Replaying at 64/32 bounds the
            # tile scratch while preserving the released operator order and
            # is safer than failing a request that can run more slowly.
            return True
        if (
            self._hot_kv_persist is None
            and self.cfg.model_type == "glm_moe_dsa"
            and self.rc.layer_stationary_prefill
            and self.rc.glm_dsa_mla_kv_spill_dir
        ):
            # Full GLM's explicit F75 route has query-tile memory proportional
            # to L*K*heads*value_width. A refusal before sampling can replay
            # exactly at 8 -> 4 -> 2 -> 1 without changing model state bytes.
            return True
        if self._hot_kv_persist is None and self.rc.adaptive_chunk_size:
            # Any F68 adaptive-chunk model (gpt_oss, GLM, Kimi K3, ...) can
            # hit a MemoryError from _fetch_experts' independent
            # governor.reserve() call even when the adaptive controller's
            # own compute-scratch budget judged the chunk safe -- see the
            # comment at this method's retry call site. Live-confirmed
            # 2026-07-29 (EpistemeAI/VibeCoder-20B, real gpt_oss checkpoint,
            # tight real memory conditions): an uncaught MemoryError from
            # this exact path killed a whole prefill request outright, with
            # no retry coverage before this fix (gpt_oss is not qwen3_5/
            # qwen3_5_moe, the only families this method previously
            # covered). Same "unstarted prefill, safe to discard and replay
            # slower" invariant applies regardless of model family.
            return True
        return bool(
            self._hot_kv_persist is None
            and self.rc.hot_prompt_kv
            and self.rc.quant_bits
            and self.cfg.model_type in ("qwen2", "qwen3")
        )

    def _select_prefill_chunk_size(self, matched_slot: "_HotPromptSlot | None") -> int:
        """F95: the actual per-conversation adaptivity decision.

        matched_slot is the hot-KV slot this request is CONTINUING (already
        popped off the LRU by the caller), or None for a brand-new
        conversation with no match at all. A continuing conversation MUST
        reuse whatever chunk size actually built that slot's KV/recurrent
        state -- hot_prompt_kv's fixed-chunk invariant applies per reuse
        lineage, not engine-wide, so reading matched_slot.chunk_size (not
        self.rc.hot_prompt_kv_chunk_size) is what makes two DIFFERENT
        conversations free to use different chunk sizes without
        conflicting. A brand-new conversation samples live memory THIS
        INSTANT and picks a size that only has to stay valid for its own
        lifetime, not this engine's -- the reason the wider ladder tiers
        (512/128) are safe again despite being unsafe when picked once per
        engine (see hybrid_prefill_chunk_size's docstring)."""
        if matched_slot is not None:
            selected = (getattr(matched_slot, "chunk_size", 0)
                        or self.rc.hot_prompt_kv_chunk_size)
        else:
            selected = hybrid_prefill_chunk_size(
                psutil.virtual_memory().available)
        # A failed partial prefill has already released any matched endpoint;
        # keep the serving wrapper's strictly lower retry rung from being
        # overwritten by a post-cleanup memory sample that looks healthy again.
        retry_ceiling = int(getattr(
            self, "_hybrid_retry_chunk_ceiling", 0) or 0)
        if retry_ceiling:
            selected = min(selected, retry_ceiling)
        return self._apply_qwen35_prefill_chunk_ceiling(selected)

    def _apply_qwen35_prefill_chunk_ceiling(self, selected: int) -> int:
        """Clamp every Qwen prefill route, including durable persistence.

        The per-conversation selector is skipped when a durable store owns a
        fixed cache lineage. Compute tiling is not part of the mixed-depth
        snapshot representation, so the operator ceiling must also be applied
        at the common execution point instead of only inside that selector.
        """
        selected = max(0, int(selected))
        configured_ceiling = int(getattr(
            self.rc, "qwen35_prefill_chunk_ceiling", 0) or 0)
        if configured_ceiling:
            selected = min(selected, configured_ceiling)
        return selected

    def _prefill_admission_positions(self, remaining_positions: int) -> int:
        """Return the imminent prefill width used for hot-KV admission.

        Admission runs before a new Qwen conversation finalizes its adaptive
        chunk selection.  The explicit operator ceiling is nevertheless
        already authoritative: charging a stale 128/512-position transient
        while execution is capped at 32 rejected the 16K gate before its
        first allocation.  Apply the same ceiling here so admission and the
        later sweep describe one physical shape.
        """
        remaining = max(1, int(remaining_positions))
        selected = max(1, int(self.rc.prefill_chunk_size or 1))
        if self.cfg.model_type in ("qwen3_5", "qwen3_5_moe"):
            selected = max(
                1, self._apply_qwen35_prefill_chunk_ceiling(selected))
        return min(selected, remaining)

    def _new_hot_prompt_slot(
            self, *, recurrent_exact_only: bool, boundary_fork_kv,
            boundary_fork_tokens: int, tokens, full_tokens, kv, logits,
            prompt_endpoint_logits, reusable_watermark: int,
            prompt_state_approximate: bool, tool_capsules, segment_chain,
            cache_namespace: str, boundary_segment_chain=()) -> "_HotPromptSlot":
        """F96: which state to retain for the next request on this lineage.

        For an ordinary (non-recurrent) model the full post-generation
        endpoint (prompt + every fed-back generated token) is a real,
        reusable artifact: a repeat, an exact-endpoint continuation, or a
        divergent branch can all match against it.

        For a recurrent_exact_only model (qwen3_5/qwen3_5_moe/kimi_linear)
        it is NOT: the released chat template re-renders any but the latest
        assistant turn without its own generation scaffold once a further
        turn follows it, so a slot retained at the full endpoint can only
        ever match a byte-identical verbatim replay of the same request --
        live-reproduced 2026-07-22 as a 100% hot-KV miss rate on every real
        second turn. When a boundary fork was taken during this request's
        own prefill (this conversation's content with NO generation
        scaffold at all -- see the fork site earlier in generate()),
        retaining THAT instead is what makes the "extension" arm of the
        matching loop above actually fire on this conversation's next turn:
        that fork's tokens are exactly what ANY future continuation of this
        conversation is guaranteed to re-render byte-identically.

        boundary_fork_kv is None whenever no fork applies (non-recurrent
        model, no stable-boundary hint was available, or this request's
        boundary already fell inside an already-matched prefix) -- degrades
        to the ordinary full-endpoint slot exactly as before this feature.
        """
        if recurrent_exact_only and boundary_fork_kv is not None:
            return _HotPromptSlot(
                tokens=tuple(tokens[:boundary_fork_tokens]),
                kv=boundary_fork_kv,
                logits=None,
                prompt_length=boundary_fork_tokens,
                prompt_logits=None,
                reusable_prefix=0,
                chunk_size=self.rc.prefill_chunk_size,
                exact_hidden=None,
                kimi_k3_prefill_schedule=getattr(
                    self, "_active_k3_prefill_schedule", ""),
                approximate=prompt_state_approximate,
                tool_capsules=(),
                segment_chain=tuple(boundary_segment_chain),
                cache_namespace=cache_namespace,
            )
        return _HotPromptSlot(
            tokens=full_tokens,
            kv=kv,
            logits=logits,
            prompt_length=len(tokens),
            prompt_logits=prompt_endpoint_logits,
            reusable_prefix=reusable_watermark,
            # F95: whatever chunk size actually built this state -- either
            # the matched slot's own value (continuing) or a fresh
            # per-conversation pick (new), set earlier in this same
            # generate() call.
            chunk_size=self.rc.prefill_chunk_size,
            exact_hidden=(
                getattr(self, "_h_last", None)
                if getattr(getattr(self, "cfg", None), "model_type", "")
                == "qwen4_exp" else None),
            kimi_k3_prefill_schedule=getattr(
                self, "_active_k3_prefill_schedule", ""),
            approximate=prompt_state_approximate,
            tool_capsules=tool_capsules,
            segment_chain=segment_chain,
            cache_namespace=cache_namespace,
        )

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
            # F95: this path already excludes durable persistence (see the
            # bail-out above), so it's always the in-memory adaptive case --
            # whatever chunk size was actually driving this interrupted
            # request is what a retry must resume with.
            chunk_size=self.rc.prefill_chunk_size,
            kimi_k3_prefill_schedule=getattr(
                self, "_active_k3_prefill_schedule", ""),
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
        # An in-flight exact expert fetch can still consult the governor and
        # cache. Join it before closing either dependency.
        if self._expert_batch_executor is not None:
            self._expert_batch_executor.shutdown(wait=True, cancel_futures=True)
            self._expert_batch_executor = None
        if self.governor is not None:
            self.governor.close()
        if self.prefetcher:
            self.prefetcher.close()
        if self.predictor is not None:
            self.predictor.save()
        reranked_exact = getattr(self._lm_head_w, "exact", None)
        if (reranked_exact is not None
                and reranked_exact is not self._streamed_lm_head
                and callable(getattr(reranked_exact, "close", None))):
            reranked_exact.close()
        if self._streamed_lm_head is not None:
            self._streamed_lm_head.close()
        if self._embed_rows is not None:
            self._embed_rows.close()
        if self._qwen4_ple_rows is not None:
            self._qwen4_ple_rows.close()
            self._qwen4_ple_rows = None
        close_store = getattr(self.store, "close", None)
        if close_store is not None:
            close_store()
        self.cache.clear()
