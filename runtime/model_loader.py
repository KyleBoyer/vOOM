"""WeightStore: lazy, per-tensor access to (possibly sharded) safetensors checkpoints.

`mx.load` on a safetensors file returns *lazy* file-backed arrays; nothing is read
until `mx.eval`. Opening a shard costs ~3 ms, so the store re-opens shards on every
fetch rather than holding evaluated arrays — residency is entirely the caller's
(or the WeightCache's) responsibility. Dropping the returned arrays is eviction.
"""

from __future__ import annotations

from bisect import bisect_left
from contextlib import contextmanager
import hashlib
import json
import os
import re
import struct
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from .config import ModelConfig
from .checkpoint_identity import (
    checkpoint_release_revision,
    has_checkpoint_receipt,
    refuse_incomplete_checkpoint,
    validate_raw_fast_tier_binding,
)
from .local_config import get_storage_config


_QUANT_MODES = {"affine", "mxfp4", "nvfp4", "mxfp8"}
_K3_EXPERT_SCALE_RE = re.compile(
    r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\."
    r"(\d+)\.(w1|w2|w3)\.weight_scale$"
)
_LAYER_PARAM_RE = re.compile(r"^model\.layers\.(\d+)\.")
_RAW_FAST_TIER_MAX_RUN_BYTES = 128_000_000


@dataclass(frozen=True)
class _QuantAux:
    scales: str
    biases: str | None
    bits: int
    group_size: int
    mode: str


@dataclass(frozen=True)
class _CTInt4Aux:
    """F93: a vllm-project/compressed-tensors "pack-quantized" INT4 triplet
    (.weight_packed/.weight_scale/.weight_shape in place of a plain
    .weight). Distinct scheme from this project's own QTensor/mx.quantize
    format above -- different library, different bit layout, dequantized
    eagerly to a dense bf16 array at fetch time (see WeightStore.fetch),
    not lazily matmul'd via mx.quantized_matmul like a QTensor."""

    packed: str
    scale: str
    shape: str


@dataclass(frozen=True)
class _DSV4Aux:
    """F213: DeepSeek V4's ``<stem>.weight`` + ``<stem>.scale`` pair.

    Two schemes share this shape and are told apart by the weight's dtype at
    fetch time, not by config: routed experts are INT8 while attention and
    shared experts are FP8 E4M3. Both carry E8M0 block scales, but with
    different blocking, so the block size is derived per tensor.
    """

    weight: str
    scale: str


@dataclass(frozen=True)
class _FineGrainedFP8Aux:
    """HF fine-grained FP8 ``weight`` + exact dequant-multiplier pair.

    Official GLM stores the multiplier grid as FP32. Some published Qwen4-Exp
    FP8 derivatives store the same values as BF16; the fetch path widens those
    exact BF16 values to FP32 before executing the unchanged decoder.
    """

    weight: str
    scale: str
    block_shape: tuple[int, int]


@dataclass(frozen=True)
class _CTMXFP4Aux:
    """F128: Kimi K3's real "mxfp4-pack-quantized" compressed-tensors pair
    (.weight_packed/.weight_scale, no .weight_shape -- confirmed by directly
    inspecting a real downloaded K3 shard; unlike K2.5's INT4 triplet, MXFP4
    ships no shape tensor at all). The conservative path dequantizes eagerly
    to a dense array at fetch time. The opt-in native path instead exposes the
    same bytes as an MLX QTensor after a zero-copy uint8-to-uint32 view; both
    representations are materialized on the fetch thread before returning."""

    packed: str
    scale: str


@dataclass(frozen=True)
class _Qwen4FusedExpertSlice:
    """One exact logical expert matrix inside Qwen4-Exp's fused BF16 body."""

    shard: str
    physical_name: str
    dtype: str
    shape: tuple[int, int]
    offset: int
    nbytes: int


def _quant_params(value) -> tuple[int, int, str] | None:
    """Normalize one standard-MLX quantization descriptor."""
    if not isinstance(value, dict):
        return None
    try:
        bits = int(value["bits"])
        group_size = int(value["group_size"])
    except (KeyError, TypeError, ValueError):
        return None
    mode = str(value.get("mode", "affine"))
    valid = (
        mode == "affine" and group_size in (32, 64, 128)
        and bits in (2, 3, 4, 5, 6, 8)
    ) or (
        (mode, group_size, bits)
        in {("mxfp4", 32, 4), ("mxfp8", 32, 8), ("nvfp4", 16, 4)}
    )
    if not valid:
        return None
    return bits, group_size, mode


def _undo_llama_cpp_gguf_rope_permute(
        weights: mx.array, n_head: int, n_head_kv: int) -> mx.array:
    """Reverse llama.cpp's real HF-to-GGUF q_proj/k_proj row permutation.

    llama.cpp's own converter (`conversion/llama.py`'s `LlamaModel`, real
    source fetched 2026-07-28) applies `permute()` to q_proj/k_proj
    weights (and biases, when present) for EVERY Llama-family GGUF export
    -- its own rope kernel wants each head's dimension pairs interleaved,
    unlike HF's native rotate-half layout this codebase's `_attention`
    (runtime/layer_runner.py, `traditional=False`) assumes. Left
    unreversed, this measured as real, large error against the actual
    unquantized bf16 weights specifically on q_proj/k_proj (mean abs error
    ~0.014 against an original whose std is ~0.018 -- roughly 100% of
    signal, vs ~1-8% for every other tensor, matching ordinary Q4_K/Q6_K
    quantization noise) and produced fluent-but-incoherent decode output
    end to end (confirmed 2026-07-28, Llama-3-Groq-8B-Tool-Use). Verified
    to be the exact inverse of the real `permute()` via numeric round-trip
    on random data before use here -- Qwen2's own conversion module has no
    such step at all (confirmed against the real source), so this must
    stay scoped to model_type == "llama" GGUF checkpoints specifically,
    not applied generically.
    """
    if n_head_kv and n_head != n_head_kv:
        n_head = n_head_kv
    d = weights.shape[0] // n_head // 2
    return (weights.reshape(n_head, d, 2, *weights.shape[1:])
                    .swapaxes(1, 2)
                    .reshape(weights.shape))


def _read_text_retry(path: Path, attempts: int = 4) -> str:
    """F24: metadata reads on externally-hosted models survive transient
    mount drops (same failure class that killed a GLM run at config-read
    time)."""
    import os as _os
    import time as _t

    for i in range(attempts):
        try:
            return path.read_text()
        except OSError:
            if i == attempts - 1:
                raise
            remount = get_storage_config().remount_command_for(path)
            if remount:
                _os.system(remount)
            _t.sleep(5 * (2 ** i))


class WeightStore:
    def __init__(self, model_dir: str | Path, fast_dirs: list[str | Path] | None = None,
                 *, require_vpack_hashes: bool = False,
                 require_raw_weight_hashes: bool = False,
                 parallel_storage_reads: bool = False,
                 native_ct_mxfp4: bool = False,
                 kimi_k3_scale_sidecar_dir: str | Path | None = None,
                 bf16_nf12_sidecar_dir: str | Path | None = None,
                 bf16_nf12_uncached_reads: bool = False,
                 bf16_nf12_direct_linear: bool = False,
                 safetensors_offset_order: bool = False):
        """fast_dirs: optional overlay directories on faster disks, ordered
        fastest-first (split placement across N drives). Packed tensor files found
        in an earlier tier are read from there instead of the primary store —
        bytes served from a fast tier leave the slow disk's critical path."""
        self.dir = get_storage_config().resolve(model_dir)
        refuse_incomplete_checkpoint(self.dir)
        self.fast_dirs = [Path(d).expanduser() for d in (fast_dirs or [])]
        self.fast_tier_bytes = 0
        self.fast_tier_tensors = 0
        self.archive_bytes = 0
        self.parallel_storage_reads = bool(parallel_storage_reads)
        # Engine phase hint for exact raw-safetensors overlays.  The released
        # archive remains authoritative when false; decode-only tier profiles
        # use this to avoid a second device/page-cache stream during the
        # already memory-dominant long prefill, then restore it for decode.
        # This must gate BOTH Qwen4's released fused-BF16 virtual rows and the
        # ordinary raw-tensor path used by per-expert FP8 conversions.
        self.raw_fast_tier_enabled = True
        self.parallel_tier_fetches = 0
        self.parallel_tier_fast_bytes = 0
        self.parallel_tier_archive_bytes = 0
        self.parallel_tier_wall_ns = 0
        self.parallel_tier_fast_service_ns = 0
        self.parallel_tier_archive_service_ns = 0
        self.parallel_tier_hidden_ns = 0
        self.safetensors_offset_order = bool(safetensors_offset_order)
        self._safetensors_headers: dict[str, dict] = {}
        self._safetensors_header_lock = threading.Lock()
        self._qwen4_fused_expert_slices: dict[
            str, _Qwen4FusedExpertSlice] = {}
        self._qwen4_fused_physical_names: set[str] = set()
        self.qwen4_expert_layout = ""
        self.qwen4_fused_read_calls = 0
        self.qwen4_fused_read_extents = 0
        self.qwen4_fused_requested_tensors = 0
        self.qwen4_fused_read_bytes = 0
        # Direct range readers otherwise reopen the same ~100-230 files for
        # every streamed layer/token.  Cached descriptors preserve identical
        # pread semantics while removing open/fstat churn.  F_NOCACHE is a
        # separate explicit experiment: it can help out-of-core scans but may
        # hurt reuse, so it never follows from descriptor caching itself.
        self._direct_fd_lock = threading.Lock()
        self._direct_fds: dict[str, tuple[int, int]] = {}
        self._direct_fd_cache_enabled = (
            os.environ.get("VMODEL_DIRECT_FD_CACHE") == "1")
        self._direct_fd_nocache = (
            os.environ.get("VMODEL_DIRECT_IO_NOCACHE") == "1")
        self.direct_fd_opens = 0
        self.direct_fd_hits = 0
        self.direct_fd_closes = 0
        self.direct_fd_open_ns = 0
        self.direct_fd_nocache_applied = 0
        self.direct_pread_calls = 0
        self.direct_pread_requested_bytes = 0
        self.direct_pread_bytes = 0
        self.direct_pread_ns = 0
        self.direct_pread_short_reads = 0
        # Experimental lossless representation switch for published
        # compressed-tensors MXFP4 pairs. The format probe and runtime fetch
        # path fail closed unless the descriptor and physical dtypes match
        # MLX's native OCP MXFP4 contract exactly.
        self.native_ct_mxfp4 = bool(native_ct_mxfp4)
        self.expert_resident_bytes_per_weight = 2.0
        # Cumulative, thread-safe transform telemetry. Fetch time already
        # includes these operations; callers must treat transform time as
        # nested rather than add it to WeightCache's store time.
        self._stage_lock = threading.Lock()
        self.ct_mxfp4_transform_ns = 0
        self.ct_mxfp4_transform_calls = 0
        self.ct_mxfp4_input_bytes = 0
        self.ct_mxfp4_resident_bytes = 0
        # Fine-grained GLM FP8 always reconstructs the released BF16 compute
        # value.  The explicit native path fuses decode/scale/rounding into one
        # Metal dispatch; these counters make its real share of weight-wait
        # time and its actual coverage visible per request.
        self.native_glm53_fp8_dequant = (
            os.environ.get("VMODEL_GLM53_NATIVE_FP8_DEQUANT") == "1"
            or os.environ.get("VMODEL_QWEN4_NATIVE_FP8_DEQUANT") == "1")
        # Explicit lossless experiments: retain routed-expert E4M3 bytes and
        # F32 block multipliers in cache, then consume singleton projections
        # with an exact direct QMV. Non-singleton calls reconstruct the existing
        # BF16 carrier on demand and use the ordinary matmul. GLM and Qwen have
        # separate policy/telemetry identities even though their published
        # 128x128 fine-grained FP8 representation is the same.
        self.glm53_fp8_direct_qmv = (
            os.environ.get("VMODEL_GLM53_FP8_DIRECT_QMV") == "1")
        self.glm53_fp8_direct_qmv_decode_only = (
            os.environ.get(
                "VMODEL_GLM53_FP8_DIRECT_QMV_DECODE_ONLY", "0") == "1")
        if (self.glm53_fp8_direct_qmv_decode_only
                and not self.glm53_fp8_direct_qmv):
            raise ValueError(
                "VMODEL_GLM53_FP8_DIRECT_QMV_DECODE_ONLY requires "
                "VMODEL_GLM53_FP8_DIRECT_QMV=1")
        # The engine changes this only at a proven prefill/decode boundary.
        # Keeping the requested policy separate from its current phase avoids
        # relabeling a BF16 cache page as packed FP8 (or vice versa).
        self.glm53_fp8_direct_qmv_active = bool(
            self.glm53_fp8_direct_qmv
            and not self.glm53_fp8_direct_qmv_decode_only)
        self.qwen4_fp8_direct_qmv = (
            os.environ.get("VMODEL_QWEN4_FP8_DIRECT_QMV") == "1")
        self.qwen4_fp8_direct_qmv_decode_only = (
            os.environ.get(
                "VMODEL_QWEN4_FP8_DIRECT_QMV_DECODE_ONLY", "0") == "1")
        if (self.qwen4_fp8_direct_qmv_decode_only
                and not self.qwen4_fp8_direct_qmv):
            raise ValueError(
                "VMODEL_QWEN4_FP8_DIRECT_QMV_DECODE_ONLY requires "
                "VMODEL_QWEN4_FP8_DIRECT_QMV=1")
        self.qwen4_fp8_direct_qmv_active = bool(
            self.qwen4_fp8_direct_qmv
            and not self.qwen4_fp8_direct_qmv_decode_only)
        # The fused decoder is exact, but medium layer-stationary profiling
        # showed that dispatching it from expert-prefetch workers can contend
        # with the foreground expert GEMM on Metal.  Keep today's all-native
        # behavior by default; an explicit experiment may leave background
        # expert pages on the eager decoder while retaining native foreground
        # reconstruction.
        self.native_glm53_fp8_prefetch = (
            os.environ.get("VMODEL_GLM53_NATIVE_FP8_PREFETCH", "1") == "1")
        self.glm53_fp8_transform_ns = 0
        self.glm53_fp8_transform_calls = 0
        self.glm53_fp8_native_calls = 0
        self.glm53_fp8_input_bytes = 0
        self.glm53_fp8_resident_bytes = 0
        self.glm53_fp8_prefetch_transform_ns = 0
        self.glm53_fp8_prefetch_transform_calls = 0
        self.glm53_fp8_prefetch_native_calls = 0
        self.glm53_fp8_direct_pages = 0
        self.glm53_fp8_direct_resident_bytes = 0
        self.glm53_fp8_direct_qmv_calls = 0
        self.glm53_fp8_direct_qmv_positions = 0
        self.glm53_fp8_direct_fallback_calls = 0
        self.glm53_fp8_direct_fallback_positions = 0
        self.glm53_fp8_direct_fallback_reconstruct_ns = 0
        self.glm53_fp8_direct_fallback_reconstruct_bytes = 0
        self.qwen4_fp8_direct_pages = 0
        self.qwen4_fp8_direct_resident_bytes = 0
        self.qwen4_fp8_direct_qmv_calls = 0
        self.qwen4_fp8_direct_qmv_positions = 0
        self.qwen4_fp8_direct_fallback_calls = 0
        self.qwen4_fp8_direct_fallback_positions = 0
        self.qwen4_fp8_direct_fallback_reconstruct_ns = 0
        self.qwen4_fp8_direct_fallback_reconstruct_bytes = 0
        self.k3_scale_sidecar = None
        self.k3_scale_sidecar_read_bytes = 0
        self.k3_scale_sidecar_output_bytes = 0
        self.k3_scale_sidecar_decode_ns = 0
        self.k3_scale_sidecar_decode_calls = 0
        self._k3_scale_sidecar_request = (
            Path(kimi_k3_scale_sidecar_dir).expanduser()
            if kimi_k3_scale_sidecar_dir else None
        )
        self.bf16_nf12_sidecar = None
        self.bf16_nf12_read_bytes = 0
        self.bf16_nf12_output_bytes = 0
        self.bf16_nf12_decode_ns = 0
        self.bf16_nf12_decode_calls = 0
        self.bf16_nf12_invalidation_attempts = 0
        self.bf16_nf12_invalidation_successes = 0
        self.bf16_nf12_invalidation_failures = 0
        self._bf16_nf12_pending_invalidations: set[int] = set()
        self._bf16_nf12_sidecar_request = (
            Path(bf16_nf12_sidecar_dir).expanduser()
            if bf16_nf12_sidecar_dir else None
        )
        self.bf16_nf12_uncached_reads = bool(
            bf16_nf12_uncached_reads
        )
        self.bf16_nf12_direct_linear = bool(
            bf16_nf12_direct_linear
        )
        # Explicit opt-in, default off, per this project's anti-overfit rule:
        # the packed path is a representation change whose fused kernel
        # reassociates float32 sums, so it stays behind a flag until a broad
        # released-model corpus passes, not just the probes that found it.
        self.dsv4_native_mxfp4 = os.environ.get(
            "VMODEL_DSV4_NATIVE_MXFP4") == "1"
        # Keep trunk FP8 weights packed in the cache and widen them at use.
        # Halves what a pinned trunk layer costs, which is the whole point:
        # the pin planner sizes on disk bytes, so packed storage also makes
        # its plan match reality instead of undercounting by 1.92x.
        self.dsv4_packed_trunk = os.environ.get(
            "VMODEL_DSV4_PACKED_TRUNK") == "1"
        if (
            self.bf16_nf12_uncached_reads
            and self._bf16_nf12_sidecar_request is None
        ):
            raise ValueError(
                "bf16_nf12_uncached_reads requires a BF16 NF12 sidecar"
            )
        if (
            self.bf16_nf12_direct_linear
            and self._bf16_nf12_sidecar_request is None
        ):
            raise ValueError(
                "bf16_nf12_direct_linear requires a BF16 NF12 sidecar"
            )
        if (
            self.bf16_nf12_direct_linear
            and self.bf16_nf12_uncached_reads
        ):
            raise ValueError(
                "bf16_nf12_direct_linear is incompatible with uncached "
                "whole-layer reads"
            )
        # F128: a SECOND, distinct fast-tier mechanism from the vpack2
        # overlay above -- for a RAW (unpacked) safetensors checkpoint
        # like Kimi K3's, mirroring a deterministic (not learned-heat-
        # predicted) subset of always-touched tensors as individual raw-
        # byte files, built by formats/kimi_k3_fast_tier.py. Populated
        # lazily on first fetch() call, not here, so constructing a
        # WeightStore never does directory I/O for a model that has no
        # fast tier staged (the common case).
        self._raw_fast_tier_manifest: dict[str, dict] | None = None
        self._raw_fast_tier_root: Path | None = None
        # MLX establishes per-calling-thread resources on first evaluation.
        # Creating a fresh ThreadPoolExecutor for every streamed layer/sweep
        # therefore leaks thousands of retained worker resources during long
        # out-of-core decode and eventually reaches Darwin's thread ceiling.
        # One lazily-created worker preserves device overlap without thread
        # churn; it lives for the same lifetime as this model store.
        self._raw_fast_tier_executor = None
        # A model released GGUF-only (e.g. VibeThinker-3B's tool-calling
        # fine-tune -- no safetensors release exists at all) -- see
        # formats/gguf_reader.py. Populated below, alongside the ordinary
        # weight_map construction; None for every other checkpoint format.
        self.gguf = None
        self._gguf_pending_real_names: dict[str, str] = {}
        self.mtplx_mtp_sidecar: str | None = None
        self._mtplx_mtp_sidecar_names: frozenset[str] = frozenset()
        self._mtplx_mtp_sidecar_sha256: str | None = None
        self._mtplx_mtp_sidecar_layout: dict[str, tuple[str, tuple[int, ...], int]] = {}
        self._mtplx_mtp_exact_fast_names: frozenset[str] = frozenset()
        # Proposal-only mixed checkpoints may keep selected MTP matrices in
        # the released BF16 sidecar while reusing packed matrices from the
        # target artifact.  These names are authoritative slow-tier tensors:
        # an older all-MXFP4 raw-fast manifest must not replace them merely
        # because the logical tensor name is identical.
        self._mtp_proposal_plain_names: frozenset[str] = frozenset()
        # Optional proposal-only representation marker.  It never changes the
        # served target body; Qwen's MTP adapter uses it solely to give a
        # validated packed draft page a round-local cache identity/lifetime.
        self.mtp_proposal_representation: str | None = None
        self.config = ModelConfig.from_dir(self.dir)
        raw_config = json.loads(_read_text_retry(self.dir / "config.json"))
        text_config = raw_config.get("text_config", {})
        quantization = (
            raw_config.get("quantization")
            or raw_config.get("quantization_config")
            or (text_config.get("quantization") if isinstance(text_config, dict) else None)
            or (text_config.get("quantization_config")
                if isinstance(text_config, dict) else None)
            or {}
        )
        self.quantization: dict = dict(quantization) if isinstance(quantization, dict) else {}
        self.on_disk_quantized = False
        self.quantization_identity = "none"
        self.quantized_bytes_per_weight = 0.0
        # Physical expert payload estimate for trace/layout simulation. This is
        # deliberately separate from the materialized WeightCache size: Kimi
        # K2.5 reads released INT4+BF16-scales but currently expands to BF16 in
        # memory, while runtime quantize-on-load does the opposite (reads BF16,
        # retains a smaller QTensor).
        self.expert_storage_bytes_per_weight = 2.0

        # Store preference: vpack2 (sequential archive, coalesced reads) over vpack
        # (per-tensor files) over raw safetensors. Both packed forms are bit-exact.
        self.vpack2 = None
        self.vpack = self.dir / "weights.vpack"
        self._vpack_overlay_manifest: dict[str, str] = {}
        if (self.dir / "weights.vpack2.index.json").exists() or (self.dir / "vpack2.CURRENT").exists():
            import sys

            root = str(Path(__file__).resolve().parent.parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            from formats.packed2 import Vpack2Reader

            self.vpack2 = Vpack2Reader(self.dir, require_hashes=require_vpack_hashes)
            overlay_manifest = self.vpack / "manifest.json"
            if overlay_manifest.exists():
                self._vpack_overlay_manifest = json.loads(
                    _read_text_retry(overlay_manifest))
        self.require_vpack_hashes = require_vpack_hashes
        self.packed = self.vpack2 is not None or (self.vpack / "manifest.json").exists()
        if self.vpack2 is not None:
            self.integrity_mode = self.vpack2.integrity_mode
            self.integrity_identity = self.vpack2.integrity_mode
        elif self.packed:
            self.integrity_mode = "legacy-vpack-no-body-hash"
            self.integrity_identity = self.integrity_mode
        elif require_raw_weight_hashes:
            from .weight_integrity import verify_manifest

            digest = verify_manifest(self.dir)
            self.integrity_mode = "raw-safetensors-sha256"
            self.integrity_identity = f"raw-sha256-{digest}"
        else:
            self.integrity_mode = "raw-safetensors-stat"
            self.integrity_identity = self.integrity_mode
        if self.vpack2 is not None:
            self.weight_map = {n: "weights.vpack2" for n in self.vpack2.index}
        elif self.packed:
            self.weight_map = json.loads(_read_text_retry(self.vpack / "manifest.json"))
        else:
            index_path = self.dir / "model.safetensors.index.json"
            single = self.dir / "model.safetensors"
            gguf_candidates = sorted(self.dir.glob("*.gguf"))
            if index_path.exists():
                index_payload = json.loads(_read_text_retry(index_path))
                self.weight_map: dict[str, str] = index_payload["weight_map"]
                # MTPLX keeps the model body's standard MLX triplets in the
                # ordinary index and ships the released BF16 native-MTP block
                # as a sidecar.  Honor only an explicit, safe index pointer;
                # never scan arbitrary sibling files into a checkpoint.
                index_metadata = index_payload.get("metadata", {})
                if not isinstance(index_metadata, dict):
                    raise ValueError(
                        "model.safetensors.index.json metadata must be an object")
                proposal_representation = index_metadata.get(
                    "vmodel_mtp_proposal_representation")
                if proposal_representation is not None:
                    if proposal_representation not in {
                        "mxfp4-q4-g32",
                        "hybrid-bf16-attn-mxfp4-mlp",
                    }:
                        raise ValueError(
                            "unsupported Qwen MTP proposal representation: "
                            f"{proposal_representation!r}")
                    self.mtp_proposal_representation = proposal_representation
                    if proposal_representation == "hybrid-bf16-attn-mxfp4-mlp":
                        plain_names = index_metadata.get(
                            "vmodel_mtp_proposal_plain_names")
                        proposal_sidecar = index_metadata.get(
                            "vmodel_mtp_proposal_plain_sidecar")
                        if not (
                            isinstance(plain_names, list)
                            and plain_names
                            and len(plain_names) == len(set(plain_names))
                            and all(
                                isinstance(name, str)
                                and name.startswith("mtp.")
                                and name.endswith(".weight")
                                for name in plain_names
                            )
                            and isinstance(proposal_sidecar, str)
                            and Path(proposal_sidecar).name == proposal_sidecar
                            and proposal_sidecar.endswith(".safetensors")
                        ):
                            raise ValueError(
                                "hybrid Qwen MTP proposal metadata is invalid")
                        sidecar_path = self.dir / proposal_sidecar
                        if not sidecar_path.is_file():
                            raise FileNotFoundError(
                                "hybrid Qwen MTP proposal sidecar is missing: "
                                f"{sidecar_path}")
                        sidecar_tensors = mx.load(str(sidecar_path))
                        invalid_plain = []
                        for name in plain_names:
                            value = sidecar_tensors.get(name)
                            if (
                                value is None
                                or value.dtype != mx.bfloat16
                                or self.weight_map.get(name) != proposal_sidecar
                            ):
                                invalid_plain.append(name)
                        if invalid_plain:
                            raise ValueError(
                                "hybrid Qwen MTP plain tensors are not indexed "
                                f"released BF16 arrays: {invalid_plain[:3]}")
                        self._mtp_proposal_plain_names = frozenset(plain_names)
                sidecar = index_metadata.get("mtplx_mtp_sidecar")
                if sidecar is not None and proposal_representation is not None:
                    raise ValueError(
                        "Qwen MTP checkpoint cannot select both a released "
                        "BF16 sidecar and a packed proposal representation")
                if sidecar is not None:
                    if not (
                        isinstance(sidecar, str)
                        and sidecar
                        and Path(sidecar).name == sidecar
                        and not Path(sidecar).is_absolute()
                        and sidecar.endswith(".safetensors")
                    ):
                        raise ValueError(
                            "unsafe MTPLX MTP sidecar path in weight index: "
                            f"{sidecar!r}")
                    sidecar_path = self.dir / sidecar
                    if not sidecar_path.is_file():
                        raise FileNotFoundError(
                            f"MTPLX MTP sidecar is missing: {sidecar_path}")
                    sidecar_tensors = mx.load(str(sidecar_path))
                    sidecar_names = [
                        name for name in sidecar_tensors
                        if name.startswith("mtp.")
                    ]
                    if not sidecar_names:
                        raise ValueError(
                            f"MTPLX MTP sidecar has no mtp.* tensors: {sidecar_path}")
                    non_bf16 = [
                        f"{name}:{sidecar_tensors[name].dtype}"
                        for name in sidecar_names
                        if sidecar_tensors[name].dtype != mx.bfloat16
                    ]
                    if non_bf16:
                        raise ValueError(
                            "MTPLX MTP sidecar must preserve released BF16 "
                            f"mtp.* tensors, found {non_bf16[:3]}")
                    collisions = sorted(set(sidecar_names) & set(self.weight_map))
                    if collisions:
                        raise ValueError(
                            "MTPLX MTP sidecar collides with indexed tensors: "
                            f"{collisions[:3]}")
                    self.weight_map.update(
                        {name: sidecar for name in sidecar_names})
                    self.mtplx_mtp_sidecar = sidecar
                    self._mtplx_mtp_sidecar_names = frozenset(sidecar_names)
                    sidecar_sha = index_metadata.get(
                        "mtplx_mtp_sidecar_sha256")
                    if isinstance(sidecar_sha, str) and len(sidecar_sha) == 64:
                        self._mtplx_mtp_sidecar_sha256 = sidecar_sha
                    self._mtplx_mtp_sidecar_layout = {
                        name: (
                            "BF16",
                            tuple(int(value) for value in sidecar_tensors[name].shape),
                            int(sidecar_tensors[name].nbytes),
                        )
                        for name in sidecar_names
                    }
            elif not single.exists() and gguf_candidates:
                # VibeThinker-3B's tool-calling fine-tune ships GGUF-only.
                # Tensor names use llama.cpp's own naming scheme
                # (blk.N.attn_q.weight, ...); canonicalize to this engine's
                # HF-style names the same way Qwen3-VL/K2.5's
                # language_model.* prefix is rewritten just below, via
                # self._real_name (populated after that loop runs, since it
                # re-initializes the dict fresh).
                if len(gguf_candidates) > 1:
                    raise ValueError(
                        f"multiple .gguf files in {self.dir}, expected exactly one: "
                        f"{[p.name for p in gguf_candidates]}")
                from formats.gguf_reader import (
                    GGUFFile, canonicalize_llama_cpp_tensor_name)

                self.gguf = GGUFFile(gguf_candidates[0])
                self.weight_map = {}
                for real_name in self.gguf.tensors:
                    canon = canonicalize_llama_cpp_tensor_name(real_name)
                    if canon is None:
                        continue
                    self.weight_map[canon] = gguf_candidates[0].name
                    self._gguf_pending_real_names[canon] = real_name
            else:
                self.weight_map = {name: single.name for name in mx.load(str(single))}

        # Qwen3-VL-class checkpoints nest the text model under
        # model.language_model.*: expose canonical model.* aliases so the
        # dense engine runs unchanged. visual.* names pass through untouched
        # (the vision tower addresses them explicitly).
        # MTPLX's self-contained multimodal layout uses the equivalent
        # vision_tower.* prefix; expose it as model.visual.* for qwen3vl.py.
        # F93 (2026-07-19): Kimi K2.5 uses the OPPOSITE order,
        # language_model.model.*, not model.language_model.* -- confirmed
        # against the real downloaded checkpoint's model.safetensors.index.json.
        # Both prefixes canonicalize to the same "model.*" the rest of the
        # engine already expects.
        #
        # 2026-07-19 (later): K2.5's lm_head is NOT nested under
        # language_model.model.* at all -- it's a sibling at
        # language_model.lm_head.weight (the wrapper's own top-level
        # attribute, not inside the inner text-model submodule). The two
        # branches above never matched it, so store.has("lm_head.weight")
        # was silently False for K2.5 -- engine.py's _lm_head_weight()
        # fallback (self.cache.get("lm_head", ["lm_head.weight"])) would
        # have raised a KeyError the first time a real request reached
        # final-logit computation (never yet exercised: every K2.5 test so
        # far failed on a memory-governor rejection during layer streaming,
        # long before logits). Generalized: strip "language_model." for ANY
        # top-level key under it, not just the ".model." submodule case.
        self._real_name: dict[str, str] = {}
        for n in list(self.weight_map):
            if n.startswith("model.language_model."):
                canon = "model." + n[len("model.language_model."):]
            elif n.startswith("language_model.model."):
                canon = "model." + n[len("language_model.model."):]
            elif n.startswith("language_model."):
                canon = n[len("language_model."):]
            elif n.startswith("vision_tower."):
                canon = "model.visual." + n[len("vision_tower."):]
            else:
                continue
            self._real_name[canon] = n
            self.weight_map[canon] = self.weight_map.pop(n)
        self._real_name.update(self._gguf_pending_real_names)

        # F213: DeepSeek V4 ships no ``model.`` prefix at all -- its tensors
        # are ``layers.N.*``, ``embed.weight``, ``head.weight``,
        # ``norm.weight``. Alias them onto the canonical names the engine
        # addresses so every scheduler/paging call site works unchanged. Only
        # applied when the canonical name is genuinely absent.
        if ("model.embed_tokens.weight" not in self.weight_map
                and "embed.weight" in self.weight_map):
            for source, canonical in (
                    ("embed.weight", "model.embed_tokens.weight"),
                    ("head.weight", "lm_head.weight"),
                    ("norm.weight", "model.norm.weight"),
                    ("hc_head_fn", "model.hc_head_fn"),
                    ("hc_head_scale", "model.hc_head_scale"),
                    ("hc_head_base", "model.hc_head_base")):
                if source in self.weight_map:
                    self.weight_map[canonical] = self.weight_map[source]
                    self._real_name[canonical] = self._real_name.get(
                        source, source)
            for name in [n for n in self.weight_map if n.startswith("layers.")]:
                canonical = "model." + name
                if canonical not in self.weight_map:
                    self.weight_map[canonical] = self.weight_map[name]
                    self._real_name[canonical] = self._real_name.get(
                        name, name)

        # F202: LFM2 names its final norm ``model.embedding_norm.weight``; the
        # engine addresses every architecture's final norm as
        # ``model.norm.weight``. Alias it only when the canonical name is
        # genuinely absent, so a checkpoint shipping both keeps its own.
        if ("model.norm.weight" not in self.weight_map
                and "model.embedding_norm.weight" in self.weight_map):
            self.weight_map["model.norm.weight"] = self.weight_map[
                "model.embedding_norm.weight"]
            self._real_name["model.norm.weight"] = self._real_name.get(
                "model.embedding_norm.weight", "model.embedding_norm.weight")

        # Official Qwen3.8 Flash-Next stores routed experts in two enormous
        # fused BF16 tensors. Published FP8 derivatives may instead expose one
        # weight+scale pair per expert matrix. Detect and validate either exact
        # physical layout before the ordinary expert pager addresses it.
        if self.config.model_type == "qwen4_exp":
            self._configure_qwen4_expert_layout()

        # Standard MLX quantized checkpoints store one logical matrix as
        # ``name.weight`` plus row/group metadata in ``name.scales`` and,
        # for affine quantization, ``name.biases``. Expose only the logical
        # matrix to the scheduler and remember which physical tensors must be
        # fetched together to reconstruct a QTensor.
        self._quant_aux: dict[str, _QuantAux] = {}
        quant_aux_names: set[str] = set()
        global_params = _quant_params(self.quantization)
        for name in list(self.weight_map):
            if not name.endswith(".weight"):
                continue
            stem = name[:-len(".weight")]
            scales = f"{stem}.scales"
            biases = f"{stem}.biases"
            if scales in self.weight_map:
                real_stem = self._real_name.get(name, name)[:-len(".weight")]
                configured = self.quantization.get(
                    stem, self.quantization.get(real_stem, None))
                params = _quant_params(configured) or global_params
                if params is None:
                    raise ValueError(
                        f"standard MLX quantized tensor {name!r} has scales but "
                        "no usable bits/group_size descriptor in config.json"
                    )
                bits, group_size, mode = params
                bias_name = biases if biases in self.weight_map else None
                self._quant_aux[name] = _QuantAux(
                    scales, bias_name, bits, group_size, mode)
                quant_aux_names.add(scales)
                if bias_name is not None:
                    quant_aux_names.add(bias_name)

        # F93: vllm-project/compressed-tensors "pack-quantized" INT4 --
        # confirmed on Kimi K2.5's real checkpoint, MoE expert weights only
        # (attention/router stay plain bf16 .weight tensors). Expose only
        # the logical "<stem>.weight" name; fetch() dequantizes the
        # packed/scale/shape triplet to a dense array (see _CTInt4Aux).
        self._ct_int4_aux: dict[str, _CTInt4Aux] = {}
        if not self.packed:
            for name in list(self.weight_map):
                if not name.endswith(".weight_packed"):
                    continue
                stem = name[:-len(".weight_packed")]
                scale = f"{stem}.weight_scale"
                shape = f"{stem}.weight_shape"
                if scale not in self.weight_map or shape not in self.weight_map:
                    continue
                logical = f"{stem}.weight"
                self._ct_int4_aux[logical] = _CTInt4Aux(name, scale, shape)
                self.weight_map[logical] = self.weight_map[name]
                quant_aux_names.add(name)
                quant_aux_names.add(scale)
                quant_aux_names.add(shape)

        # F128: vllm-project/compressed-tensors "mxfp4-pack-quantized" --
        # confirmed on Kimi K3's real checkpoint, MoE expert weights only
        # (config.json's quantization_config.ignore excludes self_attn).
        # Distinguished from K2.5's INT4 triplet above by the ABSENCE of a
        # .weight_shape tensor (K3 ships none at all; confirmed by directly
        # inspecting a real downloaded shard) rather than by dtype, which
        # would require opening a shard header per candidate at index time.
        # Logical shape is inferred at fetch time from packed.shape[1]*2,
        # safe because MXFP4 always packs exactly 2 FP4 values per uint8 byte.
        self._ct_mxfp4_aux: dict[str, _CTMXFP4Aux] = {}
        if not self.packed:
            for name in list(self.weight_map):
                if not name.endswith(".weight_packed") or name in quant_aux_names:
                    continue
                stem = name[:-len(".weight_packed")]
                scale = f"{stem}.weight_scale"
                shape = f"{stem}.weight_shape"
                if scale not in self.weight_map or shape in self.weight_map:
                    continue
                logical = f"{stem}.weight"
                self._ct_mxfp4_aux[logical] = _CTMXFP4Aux(name, scale)
                self.weight_map[logical] = self.weight_map[name]
                quant_aux_names.add(name)
                quant_aux_names.add(scale)

        # F213: DeepSeek V4 pairs every quantized matrix with a sibling
        # ``.scale``. Registering them here keeps the scheduler addressing one
        # logical ``.weight`` name while the fetch path joins the pair.
        self._dsv4_aux: dict[str, _DSV4Aux] = {}
        if (not self.packed
                and str(self.config.model_type) == "deepseek_v4"):
            for name in list(self.weight_map):
                if not name.endswith(".weight") or name in quant_aux_names:
                    continue
                scale = name[:-len(".weight")] + ".scale"
                if scale not in self.weight_map:
                    continue
                self._dsv4_aux[name] = _DSV4Aux(name, scale)
                quant_aux_names.add(scale)

        # Both released GLM-5.3 checkpoints and supported Qwen4-Exp FP8
        # derivatives use Hugging Face's fine-grained FP8 layout. The full GLM
        # deliberately retains GLM-5.2's architecture identifier because 5.3
        # changes only post-training.
        # Its float32 ``weight_scale_inv`` is a dequant multiplier over
        # 128x128 blocks, not DeepSeek-V4's E8M0 exponent-byte ``.scale``.
        # Register a separate pair type so the two encodings can never be
        # conflated.  A released BF16 GLM-5.2 remains unaffected because it
        # has no ``weight_scale_inv`` siblings.
        self._glm53_fp8_aux: dict[str, _FineGrainedFP8Aux] = {}
        if (not self.packed
                and str(self.config.model_type) in (
                    "glm5_next", "glm_moe_dsa", "qwen4_exp")):
            block = self.quantization.get("weight_block_size")
            for name in list(self.weight_map):
                if not name.endswith(".weight") or name in quant_aux_names:
                    continue
                scale = name[:-len(".weight")] + ".weight_scale_inv"
                if scale not in self.weight_map:
                    continue
                if (not isinstance(block, (list, tuple)) or len(block) != 2
                        or any(not isinstance(v, int) or isinstance(v, bool)
                               or v <= 0 for v in block)):
                    raise ValueError(
                        "fine-grained FP8 requires a positive 2-D "
                        "weight_block_size")
                self._glm53_fp8_aux[name] = _FineGrainedFP8Aux(
                    name, scale, (int(block[0]), int(block[1])))
                quant_aux_names.add(scale)

        if self.qwen4_fp8_direct_qmv and (
            self.config.model_type != "qwen4_exp"
            or self.qwen4_expert_layout != "per-expert-fp8"
            or not self._glm53_fp8_aux
        ):
            raise ValueError(
                "VMODEL_QWEN4_FP8_DIRECT_QMV requires a Qwen4-Exp "
                "per-expert fine-grained-FP8 checkpoint")

        self.on_disk_quantized = bool(self._quant_aux)
        if self.on_disk_quantized:
            identity = {
                name: {
                    "bits": aux.bits,
                    "group_size": aux.group_size,
                    "mode": aux.mode,
                }
                for name, aux in sorted(self._quant_aux.items())
            }
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:16]
            self.quantization_identity = f"mlx-{digest}"
            self.quantized_bytes_per_weight = max(
                aux.bits / 8 + (
                    8 / aux.group_size if aux.mode == "affine"
                    else 1 / aux.group_size
                )
                for aux in self._quant_aux.values()
            )
            self.expert_storage_bytes_per_weight = (
                self.quantized_bytes_per_weight)
        elif self._ct_int4_aux:
            # compressed-tensors groupwise INT4 stores 4 payload bits/weight
            # plus one BF16 scale per group. Read the released descriptor rather
            # than hard-coding K2.5's currently observed group size 32.
            candidates = []
            for group in self.quantization.get("config_groups", {}).values():
                weights = group.get("weights", {}) if isinstance(group, dict) else {}
                try:
                    bits = int(weights["num_bits"])
                    group_size = int(weights["group_size"])
                except (KeyError, TypeError, ValueError):
                    continue
                if bits > 0 and group_size > 0:
                    candidates.append(bits / 8 + 2 / group_size)
            # An unfamiliar descriptor must not make an otherwise-decodable
            # checkpoint unloadable merely for telemetry; retain the safe BF16
            # fallback when the physical format cannot be priced here.
            if candidates:
                self.expert_storage_bytes_per_weight = max(candidates)
        elif self._ct_mxfp4_aux:
            # F128: compressed-tensors MXFP4 stores 4 payload bits/weight
            # plus one E8M0 (uint8, 1 byte) scale per group -- half the
            # scale cost of K2.5's INT4 BF16 (2-byte) scale. Read the
            # released descriptor rather than hard-coding K3's currently
            # observed group size 32.
            candidates = []
            native_params = set()
            for group in self.quantization.get("config_groups", {}).values():
                weights = group.get("weights", {}) if isinstance(group, dict) else {}
                try:
                    bits = int(weights["num_bits"])
                    group_size = int(weights["group_size"])
                except (KeyError, TypeError, ValueError):
                    continue
                if bits > 0 and group_size > 0:
                    candidates.append(bits / 8 + 1 / group_size)
                    native_params.add((
                        bits,
                        group_size,
                        str(weights.get("scale_dtype", "")),
                        bool(weights.get("symmetric", False)),
                        str(weights.get("type", "")),
                    ))
            if candidates:
                self.expert_storage_bytes_per_weight = max(candidates)
            if self.native_ct_mxfp4:
                if (
                    self.quantization.get("quant_method") != "compressed-tensors"
                    or self.quantization.get("format")
                    != "mxfp4-pack-quantized"
                    or native_params
                    != {(4, 32, "torch.uint8", True, "float")}
                ):
                    raise NotImplementedError(
                        "native compressed-tensors MXFP4 requires the published "
                        "mxfp4-pack-quantized OCP E2M1/E8M0 descriptor with "
                        "bits=4 and group_size=32"
                    )
                self.expert_resident_bytes_per_weight = max(candidates)
        elif self.config.model_type == "gpt_oss":
            # Released MXFP4 blocks plus scales/metadata. Match the existing
            # conservative resident/admission estimate used by engine.py.
            self.expert_storage_bytes_per_weight = 0.6
        elif self._glm53_fp8_aux:
            # One E4M3 byte per value plus one float32 multiplier per 128x128
            # block. The runtime widens selected matrices to BF16 on fetch,
            # so this describes storage only, not the resident page.
            block = self.quantization.get("weight_block_size", (128, 128))
            if (not isinstance(block, (list, tuple)) or len(block) != 2
                    or any(not isinstance(v, int) or isinstance(v, bool)
                           or v <= 0 for v in block)):
                raise ValueError(
                    "fine-grained FP8 requires a positive 2-D "
                    "weight_block_size")
            self.expert_storage_bytes_per_weight = (
                1.0 + 4.0 / (int(block[0]) * int(block[1])))
            direct_all_phases = (
                (self.config.model_type in ("glm_moe_dsa", "glm5_next")
                 and self.glm53_fp8_direct_qmv
                 and not self.glm53_fp8_direct_qmv_decode_only)
                or (self.config.model_type == "qwen4_exp"
                    and self.qwen4_fp8_direct_qmv
                    and not self.qwen4_fp8_direct_qmv_decode_only)
            )
            if direct_all_phases:
                self.expert_resident_bytes_per_weight = (
                    self.expert_storage_bytes_per_weight)
        elif (self.quantization and self.config.model_type != "gpt_oss"
                and not (self.quantization.get("quant_method") == "compressed-tensors"
                         and self.quantization.get("format") == "pack-quantized"
                         and self._ct_int4_aux)
                and not (self.quantization.get("quant_method") == "compressed-tensors"
                         and self.quantization.get("format") == "mxfp4-pack-quantized"
                         and self._ct_mxfp4_aux)):
            # F93/F128: vllm-project/compressed-tensors pack-quantized INT4
            # and mxfp4-pack-quantized are now supported on-disk layouts
            # (see _ct_int4_aux/_ct_mxfp4_aux above and fetch()'s
            # dequantize_compressed_tensors_int4/_mxfp4 calls) -- only
            # reaches this branch, and still raises, if the declared
            # quantization method/format isn't one of those two exact
            # combinations, or claims to be but no matching tensors were
            # actually found (config/checkpoint mismatch).
            method = self.quantization.get("quant_method", "unknown")
            standard_declared = (
                _quant_params(self.quantization) is not None
                or any(_quant_params(value) is not None
                       for value in self.quantization.values())
            )
            suspicious_scales = any(
                "scale_inv" in name or name.endswith(".weight_scale")
                for name in self.weight_map
            )
            if self._dsv4_aux or self._glm53_fp8_aux:
                # Recognized released block-FP8 pair layouts.
                pass
            elif standard_declared or method != "unknown" or suspicious_scales:
                raise NotImplementedError(
                    f"unsupported on-disk quantization layout ({method}); convert "
                    "the checkpoint to standard MLX weight/scales/biases triplets"
                )

        if self._k3_scale_sidecar_request is not None:
            if not self.native_ct_mxfp4:
                raise ValueError(
                    "Kimi K3 scale sidecars require native_ct_mxfp4=True"
                )
            if self.config.model_type not in ("kimi_k3", "kimi_linear"):
                raise ValueError(
                    "Kimi K3 scale sidecars are valid only for Kimi K3 "
                    "compressed-tensors checkpoints"
                )
            from formats.kimi_k3_scale_sidecar import KimiK3ScaleSidecar

            self.k3_scale_sidecar = KimiK3ScaleSidecar(
                self.dir, self._k3_scale_sidecar_request
            )
        if self._bf16_nf12_sidecar_request is not None:
            if self.config.model_type not in ("kimi_k3", "kimi_linear"):
                raise ValueError(
                    "the initial BF16 NF12 runtime gate is scoped to Kimi "
                    "K3/Kimi Linear checkpoints"
                )
            from formats.bf16_nf12_sidecar import BF16NF12Sidecar

            self.bf16_nf12_sidecar = BF16NF12Sidecar(
                self.dir, self._bf16_nf12_sidecar_request
            )

        self._names = sorted(
            n for n in self.weight_map
            if n not in quant_aux_names
            and n not in self._qwen4_fused_physical_names)

    # ---- name queries -------------------------------------------------

    def _released_block_fp8_aux(self, name: str):
        return self._dsv4_aux.get(name) or self._glm53_fp8_aux.get(name)

    def _join_released_block_fp8(self, names, out: dict) -> dict:
        """Join and decode registered released block-FP8 physical pairs."""
        from .deepseek_v4 import PackedExpert, PackedFP8
        from .quant import (FineGrainedFP8Tensor,
                            dequantize_deepseek_v4_fp4,
                            dequantize_deepseek_v4_fp8,
                            dequantize_finegrained_fp8,
                            dequantize_finegrained_fp8_metal)

        joined: dict = {}
        for name in names:
            dsv4 = self._dsv4_aux.get(name)
            glm53 = self._glm53_fp8_aux.get(name)
            aux = dsv4 or glm53
            if aux is None:
                joined[name] = out[name]
                continue
            weight, scale = out[aux.weight], out[aux.scale]
            if glm53 is not None:
                # The released GLM checkpoint contains no packed FP4 expert
                # payloads; accepting int8 here would double the logical width
                # and quietly execute a different model.
                if (weight.dtype != mx.uint8
                        or scale.dtype not in (mx.float32, mx.bfloat16)):
                    raise ValueError(
                        "fine-grained FP8 pair has unexpected dtypes: "
                        f"weight={weight.dtype}, scale={scale.dtype}")
                physical_input_bytes = int(weight.nbytes + scale.nbytes)
                if scale.dtype == mx.bfloat16:
                    # Widening an already-rounded BF16 value is exact. The
                    # intended dequant operation remains FP8->FP32, FP32
                    # multiply, and final BF16 rounding in both decoders.
                    scale = scale.astype(mx.float32)
                is_glm_direct = (
                    self.glm53_fp8_direct_qmv_active
                    and self.config.model_type in ("glm_moe_dsa", "glm5_next"))
                is_qwen_direct = (
                    self.qwen4_fp8_direct_qmv_active
                    and self.config.model_type == "qwen4_exp")
                direct_expert = bool(
                    (is_glm_direct or is_qwen_direct)
                    and ".mlp.experts." in name)
                if direct_expert:
                    recorder = (
                        self._record_qwen4_fp8_direct_matmul
                        if is_qwen_direct
                        else self._record_glm53_fp8_direct_matmul)
                    joined[name] = FineGrainedFP8Tensor(
                        weight,
                        scale,
                        tuple(glm53.block_shape),
                        native_fallback=self.native_glm53_fp8_dequant,
                        recorder=recorder,
                    )
                    with self._stage_lock:
                        if is_qwen_direct:
                            self.qwen4_fp8_direct_pages += 1
                            self.qwen4_fp8_direct_resident_bytes += int(
                                joined[name].nbytes)
                        else:
                            self.glm53_fp8_direct_pages += 1
                            self.glm53_fp8_direct_resident_bytes += int(
                                joined[name].nbytes)
                    continue
                is_expert_prefetch = threading.current_thread().name.startswith(
                    "vmodel-expert-batch")
                use_native = (
                    self.native_glm53_fp8_dequant
                    and (self.native_glm53_fp8_prefetch
                         or not is_expert_prefetch)
                )
                started_ns = time.perf_counter_ns()
                if use_native:
                    joined[name] = dequantize_finegrained_fp8_metal(
                        weight, scale, block_shape=glm53.block_shape)
                else:
                    joined[name] = dequantize_finegrained_fp8(
                        weight, scale, block_shape=glm53.block_shape)
                mx.eval(joined[name])
                self._record_glm53_fp8_transform(
                    elapsed_ns=time.perf_counter_ns() - started_ns,
                    input_bytes=physical_input_bytes,
                    resident_bytes=int(joined[name].nbytes),
                    native=use_native,
                    expert_prefetch=is_expert_prefetch,
                )
                continue
            # Dtype, not config, decides for DeepSeek V4: routed experts are
            # packed FP4 in int8 containers while trunk/shared weights are FP8.
            if weight.dtype == mx.int8:
                if self.dsv4_native_mxfp4:
                    joined[name] = PackedExpert(
                        weight.view(mx.uint8).view(mx.uint32), scale)
                    continue
                joined[name] = dequantize_deepseek_v4_fp4(weight, scale)
            elif self.dsv4_packed_trunk:
                joined[name] = PackedFP8(weight, scale)
                continue
            else:
                joined[name] = dequantize_deepseek_v4_fp8(weight, scale)
            mx.eval(joined[name])
        return joined

    def _record_ct_mxfp4_transform(
        self, *, elapsed_ns: int, input_bytes: int, resident_bytes: int,
    ) -> None:
        with self._stage_lock:
            self.ct_mxfp4_transform_ns += max(0, int(elapsed_ns))
            self.ct_mxfp4_transform_calls += 1
            self.ct_mxfp4_input_bytes += max(0, int(input_bytes))
            self.ct_mxfp4_resident_bytes += max(0, int(resident_bytes))

    def stage_snapshot(self) -> tuple[int, int, int, int]:
        """Return cumulative representation-transform counters atomically."""
        with self._stage_lock:
            return (
                int(self.ct_mxfp4_transform_ns),
                int(self.ct_mxfp4_transform_calls),
                int(self.ct_mxfp4_input_bytes),
                int(self.ct_mxfp4_resident_bytes),
            )

    def _record_glm53_fp8_transform(
        self, *, elapsed_ns: int, input_bytes: int, resident_bytes: int,
        native: bool, expert_prefetch: bool = False,
    ) -> None:
        with self._stage_lock:
            self.glm53_fp8_transform_ns += max(0, int(elapsed_ns))
            self.glm53_fp8_transform_calls += 1
            self.glm53_fp8_native_calls += int(bool(native))
            self.glm53_fp8_input_bytes += max(0, int(input_bytes))
            self.glm53_fp8_resident_bytes += max(0, int(resident_bytes))
            if expert_prefetch:
                self.glm53_fp8_prefetch_transform_ns += max(
                    0, int(elapsed_ns))
                self.glm53_fp8_prefetch_transform_calls += 1
                self.glm53_fp8_prefetch_native_calls += int(bool(native))

    def glm53_fp8_snapshot(self) -> tuple[int, ...]:
        with self._stage_lock:
            return (
                int(self.glm53_fp8_transform_ns),
                int(self.glm53_fp8_transform_calls),
                int(self.glm53_fp8_native_calls),
                int(self.glm53_fp8_input_bytes),
                int(self.glm53_fp8_resident_bytes),
                int(self.glm53_fp8_prefetch_transform_ns),
                int(self.glm53_fp8_prefetch_transform_calls),
                int(self.glm53_fp8_prefetch_native_calls),
            )

    def _record_glm53_fp8_direct_matmul(
        self, *, direct: bool, positions: int,
        fallback_reconstruct_ns: int = 0,
        fallback_reconstruct_bytes: int = 0,
    ) -> None:
        with self._stage_lock:
            if direct:
                self.glm53_fp8_direct_qmv_calls += 1
                self.glm53_fp8_direct_qmv_positions += max(0, int(positions))
            else:
                self.glm53_fp8_direct_fallback_calls += 1
                self.glm53_fp8_direct_fallback_positions += max(
                    0, int(positions))
                self.glm53_fp8_direct_fallback_reconstruct_ns += max(
                    0, int(fallback_reconstruct_ns))
                self.glm53_fp8_direct_fallback_reconstruct_bytes += max(
                    0, int(fallback_reconstruct_bytes))

    def glm53_fp8_direct_snapshot(self) -> tuple[int, ...]:
        with self._stage_lock:
            return (
                int(self.glm53_fp8_direct_pages),
                int(self.glm53_fp8_direct_resident_bytes),
                int(self.glm53_fp8_direct_qmv_calls),
                int(self.glm53_fp8_direct_qmv_positions),
                int(self.glm53_fp8_direct_fallback_calls),
                int(self.glm53_fp8_direct_fallback_positions),
                int(self.glm53_fp8_direct_fallback_reconstruct_ns),
                int(self.glm53_fp8_direct_fallback_reconstruct_bytes),
            )

    def _record_qwen4_fp8_direct_matmul(
        self, *, direct: bool, positions: int,
        fallback_reconstruct_ns: int = 0,
        fallback_reconstruct_bytes: int = 0,
    ) -> None:
        with self._stage_lock:
            if direct:
                self.qwen4_fp8_direct_qmv_calls += 1
                self.qwen4_fp8_direct_qmv_positions += max(0, int(positions))
            else:
                self.qwen4_fp8_direct_fallback_calls += 1
                self.qwen4_fp8_direct_fallback_positions += max(
                    0, int(positions))
                self.qwen4_fp8_direct_fallback_reconstruct_ns += max(
                    0, int(fallback_reconstruct_ns))
                self.qwen4_fp8_direct_fallback_reconstruct_bytes += max(
                    0, int(fallback_reconstruct_bytes))

    def qwen4_fp8_direct_snapshot(self) -> tuple[int, ...]:
        with self._stage_lock:
            return (
                int(self.qwen4_fp8_direct_pages),
                int(self.qwen4_fp8_direct_resident_bytes),
                int(self.qwen4_fp8_direct_qmv_calls),
                int(self.qwen4_fp8_direct_qmv_positions),
                int(self.qwen4_fp8_direct_fallback_calls),
                int(self.qwen4_fp8_direct_fallback_positions),
                int(self.qwen4_fp8_direct_fallback_reconstruct_ns),
                int(self.qwen4_fp8_direct_fallback_reconstruct_bytes),
            )

    def parallel_tier_snapshot(self) -> tuple[int, ...]:
        """Return cumulative dual-device work and overlap evidence atomically."""
        with self._stage_lock:
            return (
                int(self.parallel_tier_fetches),
                int(self.parallel_tier_fast_bytes),
                int(self.parallel_tier_archive_bytes),
                int(self.parallel_tier_wall_ns),
                int(self.parallel_tier_fast_service_ns),
                int(self.parallel_tier_archive_service_ns),
                int(self.parallel_tier_hidden_ns),
            )

    def _record_parallel_tier(
        self, *, fast_bytes: int, archive_bytes: int, wall_ns: int,
        fast_service_ns: int, archive_service_ns: int,
    ) -> None:
        # Sum(service times)-wall is the work overlapped across independent
        # devices. Clamp orchestration/timer noise rather than reporting a
        # negative hidden interval.
        hidden_ns = max(
            0, int(fast_service_ns) + int(archive_service_ns) - int(wall_ns))
        with self._stage_lock:
            self.parallel_tier_fetches += 1
            self.parallel_tier_fast_bytes += int(fast_bytes)
            self.parallel_tier_archive_bytes += int(archive_bytes)
            self.parallel_tier_wall_ns += int(wall_ns)
            self.parallel_tier_fast_service_ns += int(fast_service_ns)
            self.parallel_tier_archive_service_ns += int(archive_service_ns)
            self.parallel_tier_hidden_ns += hidden_ns

    def k3_scale_sidecar_snapshot(self) -> tuple[int, int, int, int]:
        """Return cumulative exact scale-overlay counters atomically."""
        with self._stage_lock:
            return (
                int(self.k3_scale_sidecar_read_bytes),
                int(self.k3_scale_sidecar_output_bytes),
                int(self.k3_scale_sidecar_decode_ns),
                int(self.k3_scale_sidecar_decode_calls),
            )

    def bf16_nf12_snapshot(self) -> tuple[int, int, int, int]:
        """Return cumulative exact BF16-sidecar counters atomically."""
        with self._stage_lock:
            return (
                int(self.bf16_nf12_read_bytes),
                int(self.bf16_nf12_output_bytes),
                int(self.bf16_nf12_decode_ns),
                int(self.bf16_nf12_decode_calls),
            )

    def bf16_nf12_invalidation_snapshot(self) -> tuple[int, int, int]:
        """Return deferred Darwin UBC invalidation counters atomically."""
        with self._stage_lock:
            return (
                int(self.bf16_nf12_invalidation_attempts),
                int(self.bf16_nf12_invalidation_successes),
                int(self.bf16_nf12_invalidation_failures),
            )

    def _safetensors_header(self, shard: str) -> dict:
        """Return one shard's parsed safetensors header, cached per shard.

        The header is checkpoint metadata; reading it contains no prompt-,
        route-, layer-policy-, or model-name heuristic.
        """
        with self._safetensors_header_lock:
            header = self._safetensors_headers.get(shard)
            if header is None:
                path = self.dir / shard
                fd = os.open(path, os.O_RDONLY)
                try:
                    length_raw = os.pread(fd, 8, 0)
                    if len(length_raw) != 8:
                        raise EOFError(
                            f"truncated safetensors header length: {path}"
                        )
                    length = struct.unpack("<Q", length_raw)[0]
                    raw = os.pread(fd, length, 8)
                    if len(raw) != length:
                        raise EOFError(
                            f"truncated safetensors header: {path}"
                        )
                finally:
                    os.close(fd)
                header = json.loads(raw)
                self._safetensors_headers[shard] = header
        return header

    def _safetensors_physical_offset(
        self, shard: str, canonical_name: str
    ) -> int:
        """Return one tensor's payload-relative byte offset."""
        header = self._safetensors_header(shard)
        real_name = self._real_name.get(
            canonical_name, canonical_name
        )
        metadata = header.get(real_name)
        if not isinstance(metadata, dict):
            raise KeyError(
                f"{real_name!r} missing from safetensors header {shard}"
            )
        return int(metadata["data_offsets"][0])

    def _configure_qwen4_expert_layout(self) -> None:
        """Validate and expose one supported Qwen4-Exp expert layout.

        Mixed or partial representations fail before any model tensor is
        served. The per-expert FP8 branch is metadata-only here; the generic
        fine-grained pair registration below owns dtype/scale joining.
        """
        cfg = self.config
        prefixes = [
            f"model.layers.{layer}.mlp.experts"
            for layer in range(cfg.num_hidden_layers)
        ]
        if any(name.startswith("mtp.") for name in self.weight_map):
            prefixes.append("mtp.layers.0.mlp.experts")
        fused_probes = {
            f"{prefix}.{projection}"
            for prefix in prefixes
            for projection in ("gate_up_proj", "down_proj")
        }
        per_expert_probes = {
            f"{prefix}.0.{projection}.weight"
            for prefix in prefixes
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        has_fused = any(name in self.weight_map for name in fused_probes)
        has_per_expert = any(
            name in self.weight_map for name in per_expert_probes)
        if has_fused and has_per_expert:
            raise ValueError(
                "Qwen4-Exp checkpoint mixes fused and per-expert layouts")
        if has_fused:
            self._register_qwen4_fused_expert_slices()
            self.qwen4_expert_layout = "fused-bf16"
            return
        if not has_per_expert:
            raise ValueError("Qwen4-Exp checkpoint has no supported expert layout")

        missing = []
        for prefix in prefixes:
            for expert in range(int(cfg.num_experts)):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    weight = f"{prefix}.{expert}.{projection}.weight"
                    scale = f"{prefix}.{expert}.{projection}.weight_scale_inv"
                    if weight not in self.weight_map:
                        missing.append(weight)
                    if scale not in self.weight_map:
                        missing.append(scale)
                    if len(missing) >= 4:
                        break
                if len(missing) >= 4:
                    break
            if len(missing) >= 4:
                break
        if missing:
            raise ValueError(
                "Qwen4-Exp per-expert FP8 layout is incomplete: "
                + ", ".join(missing))
        self.qwen4_expert_layout = "per-expert-fp8"

    def _register_qwen4_fused_expert_slices(self) -> None:
        """Expose released fused expert rows as ordinary logical matrices.

        The released layout is gate_up=[E,2M,H] and down=[E,H,M].  A single
        expert's gate/up/down matrices are therefore contiguous file ranges.
        Registration is metadata-only and fails closed on any unexpected
        dtype, shape, missing layer, or overlapping tensor extent.
        """
        if self.packed or self.vpack2 is not None:
            raise ValueError(
                "Qwen4-Exp fused expert slicing currently requires the "
                "released raw safetensors checkpoint")
        cfg = self.config
        experts = int(cfg.num_experts)
        width = int(cfg.moe_intermediate_size)
        hidden = int(cfg.hidden_size)
        if experts <= 0 or width <= 0 or hidden <= 0:
            raise ValueError("Qwen4-Exp fused expert geometry is incomplete")

        expert_prefixes = [
            f"model.layers.{layer}.mlp.experts"
            for layer in range(cfg.num_hidden_layers)
        ]
        has_mtp = any(name.startswith("mtp.") for name in self.weight_map)
        if has_mtp:
            expert_prefixes.append("mtp.layers.0.mlp.experts")
        for prefix in expert_prefixes:
            fused_specs = (
                (f"{prefix}.gate_up_proj", (experts, 2 * width, hidden)),
                (f"{prefix}.down_proj", (experts, hidden, width)),
            )
            metadata_by_name: dict[str, tuple[str, str, dict, int]] = {}
            for canonical, expected_shape in fused_specs:
                shard = self.weight_map.get(canonical)
                if shard is None:
                    raise ValueError(
                        f"Qwen4-Exp checkpoint lacks fused tensor {canonical}")
                real = self._real_name.get(canonical, canonical)
                header = self._safetensors_header(shard)
                metadata = header.get(real)
                if not isinstance(metadata, dict):
                    raise ValueError(
                        f"Qwen4-Exp tensor {real!r} is missing from {shard}")
                shape = tuple(int(value) for value in metadata.get("shape", ()))
                offsets = metadata.get("data_offsets")
                if (
                    metadata.get("dtype") != "BF16"
                    or shape != expected_shape
                    or not isinstance(offsets, list)
                    or len(offsets) != 2
                    or int(offsets[1]) - int(offsets[0])
                    != 2 * experts * expected_shape[1] * expected_shape[2]
                ):
                    raise ValueError(
                        f"unexpected Qwen4-Exp fused expert metadata for "
                        f"{canonical}: {metadata}")
                with (self.dir / shard).open("rb") as handle:
                    length_raw = handle.read(8)
                if len(length_raw) != 8:
                    raise EOFError(f"truncated safetensors shard {shard}")
                payload_base = 8 + struct.unpack("<Q", length_raw)[0]
                metadata_by_name[canonical] = (
                    shard, real, metadata, payload_base)
                self._qwen4_fused_physical_names.add(canonical)

            gate_name = f"{prefix}.gate_up_proj"
            down_name = f"{prefix}.down_proj"
            gate_shard, gate_real, gate_meta, gate_base = metadata_by_name[
                gate_name]
            down_shard, down_real, down_meta, down_base = metadata_by_name[
                down_name]
            gate_tensor_start = gate_base + int(gate_meta["data_offsets"][0])
            down_tensor_start = down_base + int(down_meta["data_offsets"][0])
            matrix_bytes = width * hidden * 2
            down_bytes = hidden * width * 2
            for expert in range(experts):
                expert_prefix = f"{prefix}.{expert}"
                gate_expert_start = gate_tensor_start + expert * 2 * matrix_bytes
                entries = (
                    (
                        f"{expert_prefix}.gate_proj.weight",
                        _Qwen4FusedExpertSlice(
                            gate_shard, gate_real, "BF16", (width, hidden),
                            gate_expert_start, matrix_bytes),
                    ),
                    (
                        f"{expert_prefix}.up_proj.weight",
                        _Qwen4FusedExpertSlice(
                            gate_shard, gate_real, "BF16", (width, hidden),
                            gate_expert_start + matrix_bytes, matrix_bytes),
                    ),
                    (
                        f"{expert_prefix}.down_proj.weight",
                        _Qwen4FusedExpertSlice(
                            down_shard, down_real, "BF16", (hidden, width),
                            down_tensor_start + expert * down_bytes,
                            down_bytes),
                    ),
                )
                for logical, spec in entries:
                    if logical in self.weight_map:
                        raise ValueError(
                            f"Qwen4-Exp virtual expert name collides: {logical}")
                    self.weight_map[logical] = spec.shard
                    self._qwen4_fused_expert_slices[logical] = spec

    def qwen4_fused_expert_snapshot(self) -> dict[str, int]:
        """Cumulative exact direct-row I/O counters."""
        with self._stage_lock:
            return {
                "calls": int(self.qwen4_fused_read_calls),
                "extents": int(self.qwen4_fused_read_extents),
                "requested_tensors": int(
                    self.qwen4_fused_requested_tensors),
                "bytes": int(self.qwen4_fused_read_bytes),
                "virtual_tensors": len(self._qwen4_fused_expert_slices),
            }

    def direct_io_snapshot(self) -> dict[str, int]:
        """Cumulative descriptor/range-read counters for request deltas."""
        with self._direct_fd_lock:
            return {
                "fd_opens": int(self.direct_fd_opens),
                "fd_hits": int(self.direct_fd_hits),
                "fd_closes": int(self.direct_fd_closes),
                "fd_open_ns": int(self.direct_fd_open_ns),
                "fd_cached": len(self._direct_fds),
                "fd_cache_enabled": int(self._direct_fd_cache_enabled),
                "fd_nocache_applied": int(self.direct_fd_nocache_applied),
                "pread_calls": int(self.direct_pread_calls),
                "pread_requested_bytes": int(
                    self.direct_pread_requested_bytes),
                "pread_bytes": int(self.direct_pread_bytes),
                "pread_ns": int(self.direct_pread_ns),
                "pread_short_reads": int(self.direct_pread_short_reads),
            }

    def _direct_fd(self, path: Path) -> tuple[int, int]:
        """Return a store-lifetime read-only descriptor and immutable size."""
        key = os.fspath(path)
        with self._direct_fd_lock:
            cached = (
                self._direct_fds.get(key)
                if self._direct_fd_cache_enabled else None)
            if cached is not None:
                self.direct_fd_hits += 1
                return cached
            started = time.perf_counter_ns()
            descriptor = os.open(path, os.O_RDONLY)
            try:
                size = int(os.fstat(descriptor).st_size)
                nocache = False
                if self._direct_fd_nocache:
                    from .uncached_io import set_darwin_nocache

                    nocache = set_darwin_nocache(descriptor)
            except BaseException:
                os.close(descriptor)
                raise
            if self._direct_fd_cache_enabled:
                self._direct_fds[key] = (descriptor, size)
            self.direct_fd_opens += 1
            self.direct_fd_open_ns += time.perf_counter_ns() - started
            self.direct_fd_nocache_applied += int(nocache)
            return descriptor, size

    @contextmanager
    def _direct_reader(self, path: Path):
        descriptor, size = self._direct_fd(path)
        try:
            yield descriptor, size
        finally:
            if not self._direct_fd_cache_enabled:
                os.close(descriptor)
                with self._direct_fd_lock:
                    self.direct_fd_closes += 1

    def _pread_exact(self, descriptor: int, length: int, offset: int) -> bytes:
        """Read an exact range and expose real short-read/service telemetry."""
        requested = int(length)
        started = time.perf_counter_ns()
        chunks = []
        done = 0
        calls = 0
        short_reads = 0
        while done < requested:
            chunk = os.pread(descriptor, requested - done, int(offset) + done)
            calls += 1
            if not chunk:
                raise IOError(
                    f"truncated direct read at {offset}: expected {requested}, "
                    f"got {done}"
                )
            if len(chunk) < requested - done:
                short_reads += 1
            chunks.append(chunk)
            done += len(chunk)
        raw = chunks[0] if len(chunks) == 1 else b"".join(chunks)
        elapsed = time.perf_counter_ns() - started
        with self._direct_fd_lock:
            self.direct_pread_calls += calls
            self.direct_pread_requested_bytes += requested
            self.direct_pread_bytes += len(raw)
            self.direct_pread_ns += elapsed
            self.direct_pread_short_reads += short_reads
        return raw

    def _read_qwen4_fused_expert_slices(
        self, names: Sequence[str],
    ) -> tuple[dict[str, mx.array], int]:
        """Read sorted/coalesced virtual expert matrices with exact BF16 bits."""
        from formats.packed import to_mx

        by_shard: dict[str, list[tuple[str, _Qwen4FusedExpertSlice]]] = (
            defaultdict(list))
        for name in names:
            spec = self._qwen4_fused_expert_slices.get(name)
            if spec is None:
                raise KeyError(f"unknown Qwen4-Exp virtual expert tensor {name}")
            by_shard[spec.shard].append((name, spec))

        out: dict[str, mx.array] = {}
        read_bytes = 0
        extent_count = 0
        for shard, items in by_shard.items():
            ordered = sorted(items, key=lambda item: item[1].offset)
            runs: list[list[tuple[str, _Qwen4FusedExpertSlice]]] = []
            for item in ordered:
                if runs:
                    previous = runs[-1][-1][1]
                    run_start = runs[-1][0][1].offset
                    adjacent = item[1].offset == previous.offset + previous.nbytes
                    bounded = (
                        item[1].offset + item[1].nbytes - run_start
                        <= _RAW_FAST_TIER_MAX_RUN_BYTES)
                else:
                    adjacent = bounded = False
                if runs and adjacent and bounded:
                    runs[-1].append(item)
                else:
                    runs.append([item])

            path = self.dir / shard
            with self._direct_reader(path) as (fd, file_size):
                for run in runs:
                    start = run[0][1].offset
                    end = run[-1][1].offset + run[-1][1].nbytes
                    if start < 0 or end > file_size:
                        raise IOError(
                            f"Qwen4-Exp expert extent [{start}, {end}) "
                            f"exceeds {path} ({file_size} bytes)")
                    raw = self._pread_exact(fd, end - start, start)
                    read_bytes += len(raw)
                    extent_count += 1
                    view = memoryview(raw)
                    for name, spec in run:
                        relative = spec.offset - start
                        out[name] = to_mx(
                            {"dtype": spec.dtype, "shape": spec.shape},
                            view[relative:relative + spec.nbytes],
                        )
        mx.eval(list(out.values()))
        with self._stage_lock:
            self.qwen4_fused_read_calls += 1
            self.qwen4_fused_read_extents += extent_count
            self.qwen4_fused_requested_tensors += len(names)
            self.qwen4_fused_read_bytes += read_bytes
        return out, read_bytes

    def _decode_bf16_nf12_layer(
        self, layer: int, requested_names: list[str],
    ) -> tuple[dict[str, mx.array], int]:
        if self.bf16_nf12_sidecar is None:
            return {}, 0
        entry = self.bf16_nf12_sidecar.layer_entry(layer)
        specs = {
            tensor["name"]: tensor for tensor in entry["tensors"]
        }
        direct_names = []
        if self.bf16_nf12_direct_linear:
            from .bf16_nf12_linear import direct_linear_eligible

            direct_names = [
                name for name in requested_names
                if direct_linear_eligible(specs[name])
            ]
        direct_set = set(direct_names)
        decoded_names = [
            name for name in requested_names if name not in direct_set
        ]
        output = {}
        direct_read_bytes = 0
        lazy = None
        encoded = None
        # The sidecar is one mmap-backed uint8 array. Materialize that compact
        # source once on the ordinary weight-prefetch worker, then share the
        # evaluated MLX array across every direct tensor in this cache page.
        # Evaluated arrays are safe across the worker/main-thread boundary;
        # lazy arrays are not, because MLX CPU streams are thread-local.
        direct_source = None
        if direct_names:
            path = self.bf16_nf12_sidecar.layer_path(layer)
            direct_lazy = mx.load(str(path))
            direct_source = direct_lazy.get("encoded")
            if direct_source is None:
                raise ValueError(
                    f"layer {layer}: NF12 file has no encoded tensor"
                )
            mx.eval(direct_source)
            direct_lazy.clear()
        if decoded_names and not self.bf16_nf12_uncached_reads:
            path = self.bf16_nf12_sidecar.layer_path(layer)
            lazy = mx.load(str(path))
            encoded = lazy.get("encoded")
            if encoded is None:
                raise ValueError(
                    f"layer {layer}: NF12 file has no encoded tensor"
                )
        if direct_names:
            from .bf16_nf12_linear import NF12Tensor

            for name in direct_names:
                output[name] = NF12Tensor(
                    direct_source, entry, specs[name]
                )
                direct_read_bytes += int(specs[name]["encoded_bytes"])

        physical_read_bytes = 0
        if decoded_names and self.bf16_nf12_uncached_reads:
            raw_encoded, physical_read_bytes = (
                self.bf16_nf12_sidecar.read_layer(
                    layer, uncached=True
                )
            )
            # ``mx.array(np.frombuffer(bytes))`` is lazy: if fed directly into
            # the decoder, its source node can retain the nearly-1GB Python
            # ``bytes`` object throughout layer compute. Materialize the
            # compressed input first, then drop both host references before
            # constructing the decoder graph. Unified-memory Metal still owns
            # only the exact encoded stream, and that input dies after decode.
            import numpy as np

            host_view = np.frombuffer(raw_encoded, dtype=np.uint8)
            encoded = mx.array(host_view)
            mx.eval(encoded)
            del host_view, raw_encoded
        elif decoded_names:
            physical_read_bytes = int(entry["file_bytes"])
        from .bf16_nf12_sidecar import decode_names

        started = time.perf_counter_ns()
        decoded = (
            decode_names(encoded, entry, decoded_names)
            if decoded_names else {}
        )
        decode_ns = time.perf_counter_ns() - started
        # The output is materialized by decode_layer. Drop this direct source
        # handle now, but defer Darwin UBC invalidation until WeightCache evicts
        # the decoded page. MLX's evaluated Metal output retains its source graph;
        # invalidating here can report success while the cache still owns that
        # graph/mapping, letting one-shot compressed pages accumulate during a
        # full sweep. ``release_cache_pages`` is the actual lifetime boundary.
        if encoded is not None:
            del encoded
        if lazy is not None:
            lazy.clear()
        output.update(decoded)
        output_bytes = sum(
            value.nbytes for value in decoded.values()
        )
        if self.bf16_nf12_uncached_reads:
            read_bytes = physical_read_bytes
        else:
            read_bytes = direct_read_bytes + sum(
                int(specs[name]["encoded_bytes"])
                for name in decoded_names
            )
        with self._stage_lock:
            self.bf16_nf12_read_bytes += read_bytes
            self.bf16_nf12_output_bytes += output_bytes
            self.bf16_nf12_decode_ns += decode_ns
            self.bf16_nf12_decode_calls += 1
        return output, read_bytes

    def release_cache_pages(self, names: tuple[str, ...]) -> None:
        """Invalidate evicted NF12 files only after decoded arrays lose ownership.

        Failed invalidations remain queued and are retried on later evictions.
        Failure is non-fatal because the memory governor remains authoritative.
        This hook is representation/lifetime based; it is independent of prompt
        contents, routing, tools, sampling, and layer identity.
        """
        sidecar = self.bf16_nf12_sidecar
        if sidecar is None or self.bf16_nf12_uncached_reads:
            return
        for name in names:
            match = _LAYER_PARAM_RE.match(name)
            if match is None:
                continue
            layer = int(match.group(1))
            if (
                sidecar.has_layer(layer)
                and name in sidecar.encoded_names(layer)
            ):
                self._bf16_nf12_pending_invalidations.add(layer)
        for layer in tuple(sorted(self._bf16_nf12_pending_invalidations)):
            succeeded = sidecar.invalidate_layer_cache(layer)
            with self._stage_lock:
                self.bf16_nf12_invalidation_attempts += 1
                if succeeded:
                    self.bf16_nf12_invalidation_successes += 1
                else:
                    self.bf16_nf12_invalidation_failures += 1
            if succeeded:
                self._bf16_nf12_pending_invalidations.discard(layer)

    def _decode_k3_scale_sidecars(
        self, scale_names: list[str],
    ) -> tuple[dict[str, mx.array], int]:
        """Read requested expert records and fuse each layer's scale decode."""
        if self.k3_scale_sidecar is None or not scale_names:
            return {}, 0
        by_layer: dict[int, list[tuple[str, int, str]]] = defaultdict(list)
        for name in scale_names:
            match = _K3_EXPERT_SCALE_RE.fullmatch(name)
            if match is None:
                raise ValueError(f"invalid K3 expert scale name {name!r}")
            layer, expert, projection = match.groups()
            by_layer[int(layer)].append((name, int(expert), projection))

        output: dict[str, mx.array] = {}
        physical_bytes = 0
        output_bytes = 0
        decode_ns = 0
        decode_calls = 0
        for layer, requested in by_layer.items():
            expert_ids = list(dict.fromkeys(expert for _, expert, _ in requested))
            records, read_bytes = self.k3_scale_sidecar.read_records(
                layer, expert_ids
            )
            from .kimi_k3_scale_sidecar import decode_records

            decode_started = time.perf_counter_ns()
            decoded = decode_records(
                records, self.k3_scale_sidecar.projection_shapes(layer)
            )
            decode_ns += time.perf_counter_ns() - decode_started
            for name, expert, projection in requested:
                value = decoded[(expert, projection)]
                output[name] = value
                output_bytes += value.nbytes
            physical_bytes += read_bytes
            decode_calls += 1
        with self._stage_lock:
            self.k3_scale_sidecar_read_bytes += physical_bytes
            self.k3_scale_sidecar_output_bytes += output_bytes
            self.k3_scale_sidecar_decode_ns += decode_ns
            self.k3_scale_sidecar_decode_calls += decode_calls
        return output, physical_bytes

    def layer_param_names(self, layer: int) -> list[str]:
        return self.names_with_prefix(f"model.layers.{layer}.")

    def names_with_prefix(self, prefix: str) -> list[str]:
        # `_names` is immutable and sorted after construction. Start at the
        # lexicographic insertion point, then inspect only the contiguous match
        # range instead of rescanning every tensor for every streamed MoE page.
        start = bisect_left(self._names, prefix)
        end = start
        while end < len(self._names) and self._names[end].startswith(prefix):
            end += 1
        return self._names[start:end]

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def estimate_expert_storage_page_bytes(
            self, expert_prefix: str, fallback: int) -> int:
        """Return an average physical expert page size without reading weights.

        Vpack2 already has exact compressed extent lengths in its immutable
        index, so use them. Raw/legacy stores fall back to architecture/format
        math; individual safetensors do not expose compressed expert extents.
        The planner still charges pages independently, so this is an average
        byte estimate rather than a claim that every compressed page is equal.
        """
        fallback = int(fallback)
        if self.vpack2 is None:
            return fallback
        marker = f".{expert_prefix}."
        page_bytes: dict[tuple[str, int], int] = defaultdict(int)
        for logical_name in self._names:
            if marker not in logical_name:
                continue
            layer_prefix, suffix = logical_name.split(marker, 1)
            if not layer_prefix.startswith("model.layers."):
                continue
            try:
                expert = int(suffix.split(".", 1)[0])
            except ValueError:
                continue
            aux = getattr(self, "_quant_aux", {}).get(logical_name)
            physical_names = (
                (logical_name, aux.scales, aux.biases)
                if aux is not None else (logical_name,))
            found = False
            for physical in physical_names:
                if physical is None:
                    continue
                stored = self._real_name.get(physical, physical)
                entry = self.vpack2.index.get(stored)
                if entry is None:
                    continue
                page_bytes[(layer_prefix, expert)] += int(entry["len"])
                found = True
            if not found:
                page_bytes.pop((layer_prefix, expert), None)
        if not page_bytes:
            return fallback
        return round(sum(page_bytes.values()) / len(page_bytes))

    def storage_bytes(self, names: Sequence[str]) -> int:
        """Return the on-disk bytes these tensors cost, without reading them.

        Used to size a pinned trunk prefix against a byte budget before
        anything is materialized.  Prefers the representation actually served:
        an active NF12 sidecar reports its own encoded extent, otherwise the
        safetensors header's ``data_offsets`` are authoritative.  Names whose
        size cannot be established are charged zero and reported by
        ``storage_bytes_unknown`` so a caller can refuse to plan on partial
        information rather than silently under-count.
        """
        total = 0
        for name in names:
            virtual = self._qwen4_fused_expert_slices.get(name)
            if virtual is not None:
                total += virtual.nbytes
                continue
            entry = (self._raw_fast_tier_manifest or {}).get(name)
            if entry is not None:
                total += int(entry["nbytes"])
                continue
            sidecar_bytes = self._nf12_encoded_bytes(name)
            if sidecar_bytes is not None:
                total += sidecar_bytes
                continue
            metadata = self._safetensors_entry(name)
            if metadata is None:
                continue
            start, end = (int(v) for v in metadata["data_offsets"])
            total += end - start
        return total

    def finegrained_fp8_resident_bytes(self, names: Sequence[str]) -> int:
        """Exact BF16 cache payload after GLM fine-grained FP8 widening.

        Registered FP8 matrices retain their logical checkpoint shape and
        widen to two bytes/value. Unconverted controls/norms remain in their
        released dtype, so their physical safetensors extent is already the
        resident payload. Return zero on incomplete metadata so admission can
        fail closed rather than planning from a partial layer.
        """
        if not self._glm53_fp8_aux:
            return 0
        total = 0
        for name in names:
            aux = self._glm53_fp8_aux.get(name)
            if aux is None:
                entry = self._safetensors_entry(name)
                if entry is None:
                    return 0
                start, end = (int(v) for v in entry["data_offsets"])
                total += end - start
                continue
            entry = self._safetensors_entry(aux.weight)
            scale_entry = self._safetensors_entry(aux.scale)
            if entry is None or scale_entry is None:
                return 0
            shape = entry.get("shape")
            if (not isinstance(shape, list) or not shape
                    or any(not isinstance(v, int) or v <= 0 for v in shape)):
                return 0
            elements = 1
            for extent in shape:
                elements *= extent
            total += elements * 2
        return total

    def mlx_quantized_resident_bytes(self, names: Sequence[str]) -> int:
        """Exact payload bytes of standard MLX-quantized logical tensors.

        ``_layer_names`` addresses one logical ``*.weight`` while a standard
        MLX quantized checkpoint stores that weight plus scale/bias sidecars.
        Calling :meth:`storage_bytes` on logical names alone therefore omits
        the sidecars and underestimates the resident QTensor.  For this layout
        each fetched physical array is retained directly by QTensor, so the
        safetensors payload sum is also the cache-resident byte count without
        reading model data. Return zero for any other layout or incomplete
        metadata rather than returning an optimistic partial estimate.
        """
        if not self.on_disk_quantized:
            return 0
        physical: list[str] = []
        seen: set[str] = set()
        for name in names:
            aux = self._quant_aux.get(name)
            expanded = (
                (name, aux.scales, aux.biases)
                if aux is not None else (name,)
            )
            for value in expanded:
                if value is not None and value not in seen:
                    physical.append(value)
                    seen.add(value)
        if not physical or self.storage_bytes_unknown(physical):
            return 0
        return self.storage_bytes(physical)

    def _safetensors_entry(self, name: str) -> dict | None:
        """Locate one tensor's header entry by canonical or real name.

        ``weight_map`` and the shard headers are not keyed alike: the index is
        rewritten to canonical ``model.layers.*`` names during load, while the
        header inside each shard keeps the checkpoint's own prefix (K3 ships
        ``language_model.model.layers.*``). Resolving only one of the two
        silently loses every tensor whose prefix was rewritten.
        """
        virtual = self._qwen4_fused_expert_slices.get(name)
        if virtual is not None:
            return {
                "dtype": virtual.dtype,
                "shape": list(virtual.shape),
                # Only the extent length is consumed by storage accounting;
                # retain the absolute direct-read coordinates for diagnostics.
                "data_offsets": [virtual.offset,
                                 virtual.offset + virtual.nbytes],
            }
        real = self._real_name.get(name, name)
        shard = self.weight_map.get(name)
        if shard is None:
            shard = self.weight_map.get(real)
        if shard is None:
            return None
        header = self._safetensors_header(shard)
        for candidate in (real, name):
            metadata = header.get(candidate)
            if isinstance(metadata, dict):
                return metadata
        return None

    def storage_bytes_unknown(self, names: Sequence[str]) -> list[str]:
        """Names ``storage_bytes`` could not size, so callers can fail closed."""
        unknown = []
        for name in names:
            if name in (self._raw_fast_tier_manifest or {}):
                continue
            if self._nf12_encoded_bytes(name) is not None:
                continue
            if self._safetensors_entry(name) is None:
                unknown.append(name)
        return unknown

    def _nf12_encoded_bytes(self, name: str) -> int | None:
        """Encoded size of one tensor in the active NF12 trunk sidecar."""
        sidecar = self.bf16_nf12_sidecar
        if sidecar is None:
            return None
        match = _LAYER_PARAM_RE.match(name)
        if match is None:
            return None
        layer = int(match.group(1))
        if not sidecar.has_layer(layer) or name not in sidecar.encoded_names(layer):
            return None
        for tensor in sidecar.layer_entry(layer)["tensors"]:
            if tensor["name"] == name:
                return int(tensor["encoded_bytes"])
        return None

    def is_quantized(self, name: str) -> bool:
        """Whether one logical matrix is stored as an MLX quantized triplet.

        ``on_disk_quantized`` is checkpoint-wide and therefore too coarse for
        selective artifacts such as expert-only OLMoE. Sidecars and placement
        decisions must ask about the tensor they actually consume.
        """
        return name in self._quant_aux

    def quantization_ratio(self, name: str) -> float:
        """Packed bytes divided by BF16 bytes for one logical matrix."""
        aux = self._quant_aux.get(name)
        if aux is None:
            return 1.0
        metadata = 8 if aux.mode == "affine" else 1
        return (aux.bits / 8 + metadata / aux.group_size) / 2

    def uniform_quantization_ratio(self, name_fragment: str) -> float:
        """Return a safe family ratio only when every matrix is packed.

        A partially quantized family returns 1.0. Applying one optimistic ratio
        to both packed experts and a raw router/projection would make the memory
        planner claim residency that the checkpoint cannot provide.
        """
        names = [
            name for name in self._names
            if name_fragment in name and name.endswith(".weight")
        ]
        if not names or any(name not in self._quant_aux for name in names):
            return 1.0
        return max(self.quantization_ratio(name) for name in names)

    # ---- fetching -----------------------------------------------------

    def _fetch_vpack2_fast_overlay(
        self, physical_names: list[str],
    ) -> tuple[dict[str, mx.array], list[str], int]:
        """Read exact vpack tensor copies from faster disks before the archive.

        A vpack2 archive is ideal for the external drive's sequential floor,
        while a small internal tier benefits from independently copied hot
        tensors. The copied `.vt` body is authenticated against vpack2's source
        hash on the same read that decodes it, so tiering cannot weaken the
        released-byte correctness contract.
        """
        selected, remaining = self._locate_vpack2_fast_overlay(physical_names)
        if not selected:
            return {}, remaining, 0
        decoded, nbytes = self._decode_vpack2_fast_overlay(selected)
        return self._materialize_vpack2_fast_overlay(decoded), remaining, nbytes

    def _locate_vpack2_fast_overlay(
        self, physical_names: list[str],
    ) -> tuple[list[tuple[str, Path]], list[str]]:
        """Partition a request without reading either storage device."""
        if (not self.fast_dirs or not self._vpack_overlay_manifest
                or self.vpack2 is None):
            return [], list(physical_names)
        selected: list[tuple[str, Path]] = []
        remaining: list[str] = []
        for name in physical_names:
            filename = self._vpack_overlay_manifest.get(name)
            path = next(
                (root / filename for root in self.fast_dirs
                 if filename and (root / filename).is_file()),
                None,
            )
            if path is None:
                remaining.append(name)
            else:
                selected.append((name, path))
        return selected, remaining

    def _decode_vpack2_fast_overlay(
        self, selected: list[tuple[str, Path]],
    ) -> tuple[list[tuple[str, dict, object, int]], int]:
        """Read/authenticate/decode fast files without creating MLX arrays.

        This separation is what makes cross-device overlap safe: filesystem and
        zstd/numpy work may run in a worker while all MLX materialization stays
        on the engine's calling thread.
        """
        import concurrent.futures as cf
        import struct

        from formats.packed2 import decode_body

        def read_one(item):
            name, path = item
            payload = path.read_bytes()
            if len(payload) < 8:
                raise IOError(f"truncated fast-tier tensor: {path}")
            header_len = struct.unpack("<Q", payload[:8])[0]
            header_end = 8 + header_len
            if header_end > len(payload):
                raise IOError(f"truncated fast-tier header: {path}")
            try:
                header = json.loads(payload[8:header_end])
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise IOError(f"invalid fast-tier header: {path}") from error
            entry = self.vpack2.index[name]
            if header != entry["head"]:
                raise IOError(f"fast-tier tensor metadata mismatch: {name}")
            body = self.vpack2._checked_body(name, entry, payload[header_end:])
            if len(body) != int(entry["len"]):
                raise IOError(f"fast-tier tensor extent mismatch: {name}")
            return name, header, decode_body(header, body), len(body)

        workers = min(4, len(selected))
        if workers == 1:
            decoded = [read_one(selected[0])]
        else:
            with cf.ThreadPoolExecutor(max_workers=workers) as pool:
                decoded = list(pool.map(read_one, selected))
        return decoded, sum(value[3] for value in decoded)

    @staticmethod
    def _materialize_vpack2_fast_overlay(decoded) -> dict[str, mx.array]:
        from formats.packed import to_mx

        output = {
            name: to_mx(header, raw)
            for name, header, raw, _length in decoded
        }
        mx.eval(list(output.values()))
        return output

    def _parallel_overlay_is_independent(
        self, selected: list[tuple[str, Path]],
    ) -> bool:
        """True only when every selected overlay lives off the archive device."""
        if self.vpack2 is None or not selected:
            return False
        try:
            archive_device = self.vpack2.archive.stat().st_dev
            overlay_devices = {path.stat().st_dev for _name, path in selected}
        except OSError:
            return False
        return bool(overlay_devices) and archive_device not in overlay_devices

    def _ensure_raw_fast_tier_loaded(self) -> None:
        """Lazily find and load a raw-safetensors fast-tier manifest, if any
        of self.fast_dirs has one staged for this model (see
        formats/kimi_k3_fast_tier.py). Idempotent -- safe to call every
        fetch(); after the first call this is ALWAYS a real dict (empty
        when no fast_dirs were configured, or none of them has a manifest
        for this model), never None -- fetch()'s `n in
        self._raw_fast_tier_manifest` membership check depends on that."""
        if self._raw_fast_tier_manifest is not None:
            return
        self._raw_fast_tier_manifest = {}
        for root in self.fast_dirs:
            # Callers historically supplied the parent fast-tier root, while
            # server autodiscovery supplies the already model-specific path.
            # Accept both forms so a discovered raw mirror does not become
            # ``.../<model>/<model>/fast_tier_manifest.json`` and silently
            # fall back to the slow checkpoint.
            candidate = root if (
                (root / "fast_tier_manifest.json").is_file()
                or (root / "mtp-bf16-fast.manifest.json").is_file()
            ) else root / self.dir.name
            manifest_path = candidate / "fast_tier_manifest.json"
            mtp_manifest_path = candidate / "mtp-bf16-fast.manifest.json"
            if not (manifest_path.is_file() or mtp_manifest_path.is_file()):
                continue
            manifest_bytes = (
                manifest_path.read_bytes() if manifest_path.is_file() else b"{}")
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest, dict):
                raise ValueError("raw fast-tier manifest must be an object")
            qwen4_slices = getattr(
                self, "_qwen4_fused_expert_slices", {})
            qwen4_virtual_names = [
                name for name in manifest
                if name in qwen4_slices
            ]
            qwen4_fp8_physical_names: set[str] = set()
            if getattr(self, "qwen4_expert_layout", "") == "per-expert-fp8":
                for name, aux in self._glm53_fp8_aux.items():
                    if ".mlp.experts." in name and name.startswith(
                            "model.layers."):
                        qwen4_fp8_physical_names.update((aux.weight, aux.scale))
            qwen4_fp8_names = [
                name for name in manifest
                if name in qwen4_fp8_physical_names
            ]
            if qwen4_fp8_names and len(qwen4_fp8_names) != len(manifest):
                raise ValueError(
                    "Qwen4 per-expert FP8 fast tier contains a non-expert "
                    "tensor")
            qwen4_bound_names = qwen4_virtual_names or qwen4_fp8_names
            if not qwen4_bound_names and has_checkpoint_receipt(self.dir):
                validate_raw_fast_tier_binding(
                    self.dir, candidate, manifest_bytes)
            if qwen4_bound_names:
                binding_path = candidate / "qwen4_fused_expert_fast_tier.json"
                try:
                    binding = json.loads(binding_path.read_text())
                    index_bytes = (
                        self.dir / "model.safetensors.index.json").read_bytes()
                    config_bytes = (self.dir / "config.json").read_bytes()
                except (OSError, ValueError) as error:
                    raise ValueError(
                        "Qwen4 virtual fast tier lacks a readable binding"
                    ) from error
                identity_checks = {
                    "binding": isinstance(binding, dict),
                    "schema": isinstance(binding, dict) and binding.get(
                        "schema") in (
                        "voom.qwen4-fused-expert-fast-tier.v1",
                        "voom.qwen4-per-expert-fp8-fast-tier.v1",
                        "voom.qwen4-trunk-first-fast-tier.v2",
                    ),
                    # ``path_resolver`` preserves a caller's stable symlink
                    # spelling. Builders/validators resolve it so the binding
                    # names the immutable overlay directory. Both names are
                    # acceptable only while all three content hashes match.
                    "target_model": isinstance(binding, dict) and binding.get(
                        "target_model") in {
                            self.dir.name, self.dir.resolve().name},
                    "source_index_sha256": isinstance(binding, dict)
                    and binding.get("source_index_sha256")
                    == hashlib.sha256(index_bytes).hexdigest(),
                    "source_config_sha256": isinstance(binding, dict)
                    and binding.get("source_config_sha256")
                    == hashlib.sha256(config_bytes).hexdigest(),
                    "fast_manifest_sha256": isinstance(binding, dict)
                    and binding.get("fast_manifest_sha256")
                    == hashlib.sha256(manifest_bytes).hexdigest(),
                }
                release_revision = checkpoint_release_revision(self.dir)
                if release_revision:
                    identity_checks["source_revision"] = (
                        isinstance(binding, dict)
                        and binding.get("source_revision") == release_revision
                    )
                if not all(identity_checks.values()):
                    failed = ",".join(
                        name for name, passed in identity_checks.items()
                        if not passed)
                    raise ValueError(
                        "Qwen4 virtual fast-tier source identity mismatch: "
                        f"{failed}")
                if qwen4_fp8_names:
                    if not (
                        binding.get("schema")
                        == "voom.qwen4-per-expert-fp8-fast-tier.v1"
                        and binding.get("placement") == "experts"
                        and binding.get("expert_layout") == "per-expert-fp8"
                    ):
                        raise ValueError(
                            "Qwen4 per-expert FP8 fast tier has invalid "
                            "schema or layout")
                    for name in qwen4_fp8_names:
                        entry = manifest[name]
                        metadata = self._safetensors_entry(name)
                        shard = self.weight_map.get(name)
                        filename = (
                            entry.get("file")
                            if isinstance(entry, dict) else None)
                        if metadata is None or shard is None:
                            raise ValueError(
                                "Qwen4 per-expert FP8 fast-tier tensor is "
                                f"unknown: {name}")
                        shape = tuple(
                            int(value) for value in metadata.get("shape", ()))
                        start, end = (
                            int(value) for value in metadata["data_offsets"])
                        with (self.dir / shard).open("rb") as source:
                            header_length_raw = source.read(8)
                        if len(header_length_raw) != 8:
                            raise EOFError(
                                f"truncated safetensors shard {shard}")
                        source_offset = (
                            8 + struct.unpack("<Q", header_length_raw)[0]
                            + start)
                        if not (
                            isinstance(filename, str)
                            and Path(filename).name == filename
                            and (candidate / filename).is_file()
                            and str(entry.get("dtype", "")).upper()
                            == str(metadata.get("dtype", "")).upper()
                            and tuple(int(value) for value in entry.get(
                                "shape", ())) == shape
                            and int(entry.get("nbytes", -1)) == end - start
                            and entry.get("source_file") == shard
                            and int(entry.get("source_offset", -1))
                            == source_offset
                        ):
                            raise ValueError(
                                "Qwen4 per-expert FP8 fast-tier metadata "
                                f"mismatch: {name}")
                        file_size = (candidate / filename).stat().st_size
                        offset = int(entry.get("offset", -1))
                        if offset < 0 or offset + end - start > file_size:
                            raise ValueError(
                                "Qwen4 per-expert FP8 fast-tier extent "
                                f"mismatch: {name}")
                for name in qwen4_virtual_names:
                    entry = manifest[name]
                    spec = qwen4_slices[name]
                    filename = entry.get("file") if isinstance(entry, dict) else None
                    if not (
                        isinstance(filename, str)
                        and Path(filename).name == filename
                        and (candidate / filename).is_file()
                        and str(entry.get("dtype", "")).upper() == spec.dtype
                        and tuple(int(value) for value in entry.get("shape", ()))
                        == spec.shape
                        and int(entry.get("nbytes", -1)) == spec.nbytes
                        and entry.get("source_file") == spec.shard
                        and int(entry.get("source_offset", -1)) == spec.offset
                    ):
                        raise ValueError(
                            f"Qwen4 virtual fast-tier metadata mismatch: {name}")
                    file_size = (candidate / filename).stat().st_size
                    offset = int(entry.get("offset", -1))
                    if offset < 0 or offset + spec.nbytes > file_size:
                        raise ValueError(
                            f"Qwen4 virtual fast-tier extent mismatch: {name}")
                if (
                    binding.get("schema")
                    == "voom.qwen4-trunk-first-fast-tier.v2"
                ):
                    if binding.get("placement") != "trunk-first":
                        raise ValueError(
                            "Qwen4 trunk-first fast tier has invalid placement")
                    for name, entry in manifest.items():
                        if name in qwen4_slices:
                            continue
                        if not isinstance(entry, dict):
                            raise ValueError(
                                f"Qwen4 trunk fast-tier entry is invalid: {name}")
                        metadata = self._safetensors_entry(name)
                        shard = self.weight_map.get(name)
                        filename = entry.get("file")
                        if metadata is None or shard is None:
                            raise ValueError(
                                f"Qwen4 trunk fast-tier tensor is unknown: {name}")
                        shape = tuple(
                            int(value) for value in metadata.get("shape", ()))
                        start, end = (
                            int(value) for value in metadata["data_offsets"])
                        with (self.dir / shard).open("rb") as source:
                            header_length_raw = source.read(8)
                        if len(header_length_raw) != 8:
                            raise EOFError(
                                f"truncated safetensors shard {shard}")
                        source_offset = (
                            8 + struct.unpack("<Q", header_length_raw)[0]
                            + start
                        )
                        if not (
                            isinstance(filename, str)
                            and Path(filename).name == filename
                            and (candidate / filename).is_file()
                            and str(entry.get("dtype", "")).upper()
                            == str(metadata.get("dtype", "")).upper()
                            and tuple(int(value) for value in entry.get(
                                "shape", ())) == shape
                            and int(entry.get("nbytes", -1)) == end - start
                            and entry.get("source_file") == shard
                            and int(entry.get("source_offset", -1))
                            == source_offset
                        ):
                            raise ValueError(
                                f"Qwen4 trunk fast-tier metadata mismatch: "
                                f"{name}")
                        file_size = (candidate / filename).stat().st_size
                        offset = int(entry.get("offset", -1))
                        if offset < 0 or offset + end - start > file_size:
                            raise ValueError(
                                f"Qwen4 trunk fast-tier extent mismatch: {name}")
            if getattr(self, "mtp_proposal_representation", None) is not None:
                alias_path = candidate / "mtp-quant-fast-alias.manifest.json"
                try:
                    alias_bytes = alias_path.read_bytes()
                    alias = json.loads(alias_bytes)
                    target_index_bytes = (
                        self.dir / "model.safetensors.index.json").read_bytes()
                    clone_manifest_bytes = (
                        self.dir / "mtp-quant-clone.manifest.json").read_bytes()
                    clone_manifest = json.loads(clone_manifest_bytes)
                except (OSError, ValueError) as error:
                    raise ValueError(
                        "packed Qwen MTP fast tier lacks a bound alias manifest"
                    ) from error
                if not (
                    isinstance(alias, dict)
                    and alias.get("schema")
                    == "voom.qwen-mtp-quant-fast-alias.v1"
                    and alias.get("target_model") == self.dir.name
                    and alias.get("target_index_sha256")
                    == hashlib.sha256(target_index_bytes).hexdigest()
                    and alias.get("target_clone_manifest_sha256")
                    == hashlib.sha256(clone_manifest_bytes).hexdigest()
                    and alias.get("source_manifest_sha256")
                    == hashlib.sha256(manifest_bytes).hexdigest()
                    and alias.get("target_body_mapping_sha256")
                    == clone_manifest.get("target_body_mapping_sha256")
                ):
                    raise ValueError(
                        "packed Qwen MTP fast-tier alias identity mismatch")
            if mtp_manifest_path.is_file():
                mtp_manifest = json.loads(mtp_manifest_path.read_text())
                if (
                    mtp_manifest.get("schema")
                    != "voom.qwen-mtp-bf16-fast-tier.v1"
                    or not self._mtplx_mtp_sidecar_sha256
                    or mtp_manifest.get("source_sha256")
                    != self._mtplx_mtp_sidecar_sha256
                    or mtp_manifest.get("source_sidecar")
                    != self.mtplx_mtp_sidecar
                ):
                    raise ValueError("BF16 MTP fast-tier source identity mismatch")
                fast_file = mtp_manifest.get("fast_file")
                if not (
                    isinstance(fast_file, str)
                    and Path(fast_file).name == fast_file
                    and (candidate / fast_file).is_file()
                ):
                    raise ValueError("unsafe or missing BF16 MTP fast-tier file")
                digest = hashlib.sha256()
                with (candidate / fast_file).open("rb") as source:
                    for chunk in iter(
                            lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != mtp_manifest.get("fast_file_sha256"):
                    raise ValueError("BF16 MTP fast-tier SHA-256 mismatch")
                tensors = mtp_manifest.get("tensors")
                if not isinstance(tensors, dict) or not tensors:
                    raise ValueError("BF16 MTP fast-tier has no tensor map")
                exact_names = set()
                for name, entry in tensors.items():
                    layout = self._mtplx_mtp_sidecar_layout.get(name)
                    if (
                        layout is None
                        or not isinstance(entry, dict)
                        or entry.get("file") != fast_file
                        or str(entry.get("dtype", "")).upper() != layout[0]
                        or tuple(int(value) for value in entry.get("shape", ()))
                        != layout[1]
                        or int(entry.get("nbytes", -1)) != layout[2]
                    ):
                        raise ValueError(
                            f"BF16 MTP fast-tier metadata mismatch: {name}")
                    manifest[name] = entry
                    exact_names.add(name)
                self._mtplx_mtp_exact_fast_names = frozenset(exact_names)
            self._raw_fast_tier_manifest = manifest
            self._raw_fast_tier_root = candidate
            return

    def _read_raw_fast_tier_tensors(
        self, names: list[str],
    ) -> tuple[dict[str, mx.array], int]:
        """Read raw bytes directly from the fast-tier mirror -- no shard
        file, no MLX lazy-load machinery, just the exact byte range
        formats/kimi_k3_fast_tier.py copied verbatim from the real
        checkpoint. dtype/shape come from the manifest it wrote at
        extraction time (from the SAME real safetensors header the slow
        path would have read), not re-derived here."""
        from formats.packed import to_mx

        out: dict[str, mx.array] = {}
        nbytes = 0
        by_file: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for name in names:
            entry = self._raw_fast_tier_manifest[name]
            by_file[entry["file"]].append((name, entry))

        for filename, requested in by_file.items():
            path = self._raw_fast_tier_root / filename
            if path.suffix == ".safetensors":
                lazy = mx.load(str(path))
                dtype_names = {
                    "bfloat16": "BF16",
                    "float16": "F16",
                    "float32": "F32",
                    "uint8": "U8",
                    "int8": "I8",
                    "int32": "I32",
                    "uint32": "U32",
                    "int64": "I64",
                }
                for name, entry in requested:
                    try:
                        array = lazy[name]
                    except KeyError as error:
                        raise IOError(
                            f"fast-tier container {path} is missing {name}"
                        ) from error
                    expected_dtype = str(entry["dtype"]).upper()
                    observed_dtype = dtype_names.get(
                        str(array.dtype).split(".")[-1])
                    if expected_dtype == "F8_E4M3" and observed_dtype == "U8":
                        observed_dtype = "F8_E4M3"
                    if (
                        tuple(int(value) for value in array.shape)
                        != tuple(int(value) for value in entry["shape"])
                        or observed_dtype != expected_dtype
                    ):
                        raise IOError(
                            f"fast-tier tensor metadata mismatch: {name}"
                        )
                    out[name] = array
                    nbytes += int(array.nbytes)
                mx.eval([out[name] for name, _entry in requested])
                continue

            ordered = sorted(
                requested,
                key=lambda item: int(item[1].get("offset", 0)),
            )
            runs: list[list[tuple[str, dict]]] = []
            for item in ordered:
                offset = int(item[1].get("offset", 0))
                item_end = offset + int(item[1]["nbytes"])
                if runs:
                    run_start = int(
                        runs[-1][0][1].get("offset", 0)
                    )
                    previous = runs[-1][-1][1]
                    previous_end = (
                        int(previous.get("offset", 0))
                        + int(previous["nbytes"])
                    )
                else:
                    previous_end = -1
                if (
                    runs
                    and offset == previous_end
                    and item_end - run_start
                    <= _RAW_FAST_TIER_MAX_RUN_BYTES
                ):
                    runs[-1].append(item)
                else:
                    runs.append([item])

            with self._direct_reader(path) as (fd, file_size):
                for run in runs:
                    run_start = int(run[0][1].get("offset", 0))
                    run_end = (
                        int(run[-1][1].get("offset", 0))
                        + int(run[-1][1]["nbytes"])
                    )
                    if run_start < 0 or run_end > file_size:
                        raise IOError(
                            f"fast-tier extent [{run_start}, {run_end}) "
                            f"exceeds {path} ({file_size} bytes)"
                        )
                    raw = self._pread_exact(
                        fd, run_end - run_start, run_start)
                    nbytes += len(raw)
                    view = memoryview(raw)
                    for name, entry in run:
                        relative = (
                            int(entry.get("offset", 0)) - run_start
                        )
                        length = int(entry["nbytes"])
                        tensor_raw = view[relative:relative + length]
                        out[name] = to_mx(
                            {
                                "dtype": entry["dtype"],
                                "shape": entry["shape"],
                            },
                            tensor_raw,
                        )
        mx.eval(list(out.values()))
        return out, nbytes

    def _raw_fast_tier_is_independent(self, names: list[str]) -> bool:
        """Prove selected raw-overlay files live off the checkpoint device.

        A configured directory is not enough: it may be a symlink or another
        path on the same APFS volume.  Parallel reads on one physical device
        create contention, so fail closed to the serial path unless every
        selected file resolves to a different ``st_dev``.
        """
        if not names or self._raw_fast_tier_root is None:
            return False
        try:
            source_device = self.dir.stat().st_dev
            devices = {
                (
                    self._raw_fast_tier_root
                    / self._raw_fast_tier_manifest[name]["file"]
                ).stat().st_dev
                for name in names
            }
        except (KeyError, OSError):
            return False
        return bool(devices) and source_device not in devices

    def _raw_fast_tier_executor_for_reads(self):
        """Return the store-lifetime worker used for independent-tier reads."""
        import concurrent.futures as cf

        with self._stage_lock:
            if self._raw_fast_tier_executor is None:
                self._raw_fast_tier_executor = cf.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="voom-fast-tier",
                )
            return self._raw_fast_tier_executor

    def _read_qwen4_virtual_expert_tiers(
        self, names: list[str],
    ) -> tuple[dict[str, mx.array], int]:
        """Read exact virtual expert rows across independent raw tiers."""
        self._ensure_raw_fast_tier_loaded()
        fast_names = [
            name for name in names
            if self.raw_fast_tier_enabled
            and name in (self._raw_fast_tier_manifest or {})
        ]
        fast_set = set(fast_names)
        slow_names = [name for name in names if name not in fast_set]
        out: dict[str, mx.array] = {}
        fast_bytes = 0
        slow_bytes = 0

        if (
            self.parallel_storage_reads
            and fast_names and slow_names
            and self._raw_fast_tier_is_independent(fast_names)
        ):
            executor = self._raw_fast_tier_executor_for_reads()
            parallel_started_ns = time.perf_counter_ns()

            def load_fast_timed():
                started_ns = time.perf_counter_ns()
                value = self._read_raw_fast_tier_tensors(fast_names)
                return value, time.perf_counter_ns() - started_ns

            future = executor.submit(load_fast_timed)
            archive_started_ns = time.perf_counter_ns()
            slow_out, slow_bytes = self._read_qwen4_fused_expert_slices(
                slow_names)
            archive_service_ns = time.perf_counter_ns() - archive_started_ns
            (fast_out, fast_bytes), fast_service_ns = future.result()
            wall_ns = time.perf_counter_ns() - parallel_started_ns
            out.update(slow_out)
            out.update(fast_out)
            self._record_parallel_tier(
                fast_bytes=fast_bytes,
                archive_bytes=slow_bytes,
                wall_ns=wall_ns,
                fast_service_ns=fast_service_ns,
                archive_service_ns=archive_service_ns,
            )
        else:
            if fast_names:
                fast_out, fast_bytes = self._read_raw_fast_tier_tensors(
                    fast_names)
                out.update(fast_out)
            if slow_names:
                slow_out, slow_bytes = self._read_qwen4_fused_expert_slices(
                    slow_names)
                out.update(slow_out)

        if fast_names:
            self.fast_tier_bytes += fast_bytes
            self.fast_tier_tensors += len(fast_names)
            with self._stage_lock:
                self.qwen4_fused_requested_tensors += len(fast_names)
                self.qwen4_fused_read_bytes += fast_bytes
                if not slow_names:
                    self.qwen4_fused_read_calls += 1
        self.archive_bytes += slow_bytes
        return out, fast_bytes + slow_bytes

    def close(self) -> None:
        """Join the fast-tier worker and close store-lifetime descriptors."""
        with self._stage_lock:
            executor = self._raw_fast_tier_executor
            self._raw_fast_tier_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        direct_fd_lock = getattr(self, "_direct_fd_lock", None)
        if direct_fd_lock is None:
            descriptors = []
        else:
            with direct_fd_lock:
                descriptors = [
                    item[0] for item in self._direct_fds.values()]
                self._direct_fds = {}
        for descriptor in descriptors:
            os.close(descriptor)
        if descriptors:
            with self._direct_fd_lock:
                self.direct_fd_closes += len(descriptors)

    def fetch(self, names: list[str]) -> tuple[dict[str, mx.array], float, int]:
        """Materialize tensors; return arrays, wall seconds, store-accounted bytes.

        For raw safetensors the byte field is requested logical tensor payload,
        not an OS/device measurement of physical reads or SMB traffic. Packed
        backends may account compressed extents. Callers must not label this
        field "physical bytes" without independent process/device counters.
        """
        virtual_names = [
            name for name in names
            if name in self._qwen4_fused_expert_slices]
        if virtual_names:
            started = time.perf_counter()
            virtual_out, virtual_bytes = (
                self._read_qwen4_virtual_expert_tiers(virtual_names))
            remaining = [name for name in names if name not in virtual_out]
            if remaining:
                regular_out, _regular_seconds, regular_bytes = self.fetch(
                    remaining)
                virtual_out.update(regular_out)
                virtual_bytes += regular_bytes
            return virtual_out, time.perf_counter() - started, virtual_bytes
        if self.gguf is not None:
            # Dequantize eagerly to a dense bf16 array at fetch time -- same
            # convention as the compressed-tensors INT4/MXFP4 paths below,
            # not the lazy QTensor path standard MLX quantization uses.
            t0 = time.perf_counter()
            out: dict[str, mx.array] = {}
            nbytes = 0
            for name in names:
                real_name = self._real_name.get(name, name)
                arr = self.gguf.load(real_name, out_dtype=mx.bfloat16)
                if (self.config.model_type == "llama"
                        and name.endswith(("self_attn.q_proj.weight",
                                            "self_attn.q_proj.bias"))):
                    arr = _undo_llama_cpp_gguf_rope_permute(
                        arr, self.config.num_attention_heads,
                        self.config.num_attention_heads)
                elif (self.config.model_type == "llama"
                        and name.endswith(("self_attn.k_proj.weight",
                                            "self_attn.k_proj.bias"))):
                    arr = _undo_llama_cpp_gguf_rope_permute(
                        arr, self.config.num_attention_heads,
                        self.config.num_key_value_heads)
                mx.eval(arr)
                out[name] = arr
                nbytes += arr.nbytes
            return out, time.perf_counter() - t0, nbytes
        if self.vpack2 is not None or self.packed:
            # Packed reads perform real I/O and decode inside the call. Retry the
            # whole transaction just like raw safetensors, reopening vpack2 after
            # a remount so a cycled mountpoint (e.g. Plex -> Plex-N) cannot remain stale.
            t0 = time.perf_counter()
            for attempt in range(4):
                try:
                    physical_names: list[str] = []
                    seen: set[str] = set()
                    for name in names:
                        aux = self._quant_aux.get(name)
                        released_fp8 = self._released_block_fp8_aux(name)
                        if aux is not None:
                            expanded = (name, aux.scales, aux.biases)
                        elif released_fp8 is not None:
                            expanded = (
                                released_fp8.weight, released_fp8.scale)
                        else:
                            expanded = (name,)
                        for physical_name in expanded:
                            if (physical_name is not None
                                    and physical_name not in seen):
                                physical_names.append(physical_name)
                                seen.add(physical_name)
                    if self.vpack2 is not None:
                        # The archive index retains released physical names,
                        # while multimodal wrappers are exposed to the engine
                        # through canonical model.* aliases. Translate at the
                        # archive boundary and translate results back; passing
                        # canonical Qwen3-VL/Qwen3.6/K2.5 names directly into a
                        # physical-name index otherwise raises a late KeyError.
                        physical = [self._real_name.get(name, name)
                                    for name in physical_names]
                        selected, remaining = (
                            self._locate_vpack2_fast_overlay(physical))
                        parallel_tiers = bool(
                            self.parallel_storage_reads and selected and remaining
                            and self._parallel_overlay_is_independent(selected))
                        if parallel_tiers:
                            import concurrent.futures as cf

                            # The worker returns only bytes/numpy. vpack2.fetch
                            # remains on this thread because it materializes MLX
                            # arrays after its own I/O/decode worker pool joins.
                            parallel_started_ns = time.perf_counter_ns()

                            def load_fast_overlay_timed():
                                started_ns = time.perf_counter_ns()
                                value = self._decode_vpack2_fast_overlay(selected)
                                return value, time.perf_counter_ns() - started_ns

                            with cf.ThreadPoolExecutor(max_workers=1) as pool:
                                fast_future = pool.submit(
                                    load_fast_overlay_timed)
                                archive_started_ns = time.perf_counter_ns()
                                archive, _, archive_bytes = self.vpack2.fetch(
                                    remaining)
                                archive_service_ns = (
                                    time.perf_counter_ns() - archive_started_ns)
                                (decoded, fast_bytes), fast_service_ns = (
                                    fast_future.result())
                            parallel_wall_ns = (
                                time.perf_counter_ns() - parallel_started_ns)
                            fetched = dict(archive)
                            fetched.update(
                                self._materialize_vpack2_fast_overlay(decoded))
                            self._record_parallel_tier(
                                fast_bytes=fast_bytes,
                                archive_bytes=archive_bytes,
                                wall_ns=parallel_wall_ns,
                                fast_service_ns=fast_service_ns,
                                archive_service_ns=archive_service_ns,
                            )
                        else:
                            if selected:
                                decoded, fast_bytes = (
                                    self._decode_vpack2_fast_overlay(selected))
                                fetched = self._materialize_vpack2_fast_overlay(
                                    decoded)
                            else:
                                fetched, fast_bytes = {}, 0
                            if remaining:
                                archive, _, archive_bytes = self.vpack2.fetch(
                                    remaining)
                                fetched.update(archive)
                            else:
                                archive_bytes = 0
                        nbytes = fast_bytes
                        self.fast_tier_bytes += fast_bytes
                        self.fast_tier_tensors += len(selected)
                        if remaining:
                            nbytes += archive_bytes
                            self.archive_bytes += archive_bytes
                        out = {
                            logical: fetched[stored]
                            for logical, stored in zip(physical_names, physical)
                        }
                    else:
                        out, _, nbytes = self._fetch_packed(physical_names)
                    if self._quant_aux:
                        from .quant import QTensor

                        logical = {}
                        for name in names:
                            aux = self._quant_aux.get(name)
                            if aux is None:
                                logical[name] = out[name]
                                continue
                            logical[name] = QTensor(
                                out[name], out[aux.scales],
                                (out[aux.biases]
                                 if aux.biases is not None else None),
                                aux.bits, aux.group_size, aux.mode,
                            )
                        out = logical
                    if self._dsv4_aux or self._glm53_fp8_aux:
                        out = self._join_released_block_fp8(names, out)
                    return out, time.perf_counter() - t0, nbytes
                except (OSError, RuntimeError, EOFError):
                    mx.clear_cache()
                    if attempt == 3:
                        raise
                    self._recover_nas_mount()
                    if self.vpack2 is not None:
                        from formats.packed2 import Vpack2Reader

                        self.vpack2 = Vpack2Reader(
                            self.dir, require_hashes=self.require_vpack_hashes
                        )
                    time.sleep(5 * (2 ** attempt))
        physical_names: list[str] = []
        seen: set[str] = set()
        for n in names:
            ct_aux = self._ct_int4_aux.get(n)
            ct_mxfp4_aux = self._ct_mxfp4_aux.get(n)
            if ct_aux is not None:
                # F93: the logical "<stem>.weight" name has NO physical
                # tensor of its own here (unlike _quant_aux, where the
                # logical name IS a real quantized weight tensor plus
                # sidecars) -- only packed/scale/shape physically exist.
                expanded = (ct_aux.packed, ct_aux.scale, ct_aux.shape)
            elif ct_mxfp4_aux is not None:
                # F128: same reasoning as the INT4 case above, minus the
                # shape sidecar (K3's MXFP4 pairs ship none).
                expanded = (ct_mxfp4_aux.packed, ct_mxfp4_aux.scale)
            elif self._released_block_fp8_aux(n) is not None:
                # The logical name is the physical weight tensor; its sibling
                # scale must be fetched in the same transaction.
                released_fp8 = self._released_block_fp8_aux(n)
                expanded = (released_fp8.weight, released_fp8.scale)
            else:
                aux = self._quant_aux.get(n)
                expanded = ((n, aux.scales, aux.biases) if aux is not None else (n,))
            for physical in expanded:
                if physical is not None and physical not in seen:
                    physical_names.append(physical)
                    seen.add(physical)

        # F139: an explicit, checkpoint-fingerprinted scale overlay replaces
        # only eligible expert .weight_scale tensors. Packed FP4 payloads and
        # every non-expert tensor retain the ordinary released-checkpoint path.
        # Partial generations are valid: missing layers fall back to their raw
        # safetensors scales, so a bounded gate never requires a 25GB build.
        sidecar_scale_names: list[str] = []
        if self.k3_scale_sidecar is not None:
            for physical in physical_names:
                match = _K3_EXPERT_SCALE_RE.fullmatch(physical)
                if (
                    match is not None
                    and self.k3_scale_sidecar.has_layer(int(match.group(1)))
                ):
                    sidecar_scale_names.append(physical)
        sidecar_scale_set = set(sidecar_scale_names)
        nf12_names_by_layer: dict[int, list[str]] = defaultdict(list)
        if self.bf16_nf12_sidecar is not None:
            encoded_by_layer: dict[int, set[str]] = {}
            specs_by_layer: dict[int, dict[str, dict]] = {}
            direct_linear_eligible = None
            if self.bf16_nf12_direct_linear:
                from .bf16_nf12_linear import (
                    direct_linear_eligible as _direct_linear_eligible,
                )

                direct_linear_eligible = _direct_linear_eligible
            for physical in physical_names:
                match = _LAYER_PARAM_RE.match(physical)
                if match is None:
                    continue
                layer = int(match.group(1))
                if not self.bf16_nf12_sidecar.has_layer(layer):
                    continue
                encoded = encoded_by_layer.get(layer)
                if encoded is None:
                    encoded = self.bf16_nf12_sidecar.encoded_names(layer)
                    encoded_by_layer[layer] = encoded
                if physical in encoded:
                    # Direct mode uses NF12 only where the fused consumer can
                    # actually avoid dense reconstruction. Small norms and
                    # unsupported matrix shapes are cheaper to read directly
                    # from their released BF16 tensors than to fault/map an
                    # entire layer sidecar for a tiny exact decode.
                    if direct_linear_eligible is not None:
                        specs = specs_by_layer.get(layer)
                        if specs is None:
                            specs = {
                                tensor["name"]: tensor
                                for tensor in self.bf16_nf12_sidecar
                                .layer_entry(layer)["tensors"]
                            }
                            specs_by_layer[layer] = specs
                        if not direct_linear_eligible(specs[physical]):
                            continue
                    nf12_names_by_layer[layer].append(physical)
        nf12_name_set = {
            name
            for layer_names in nf12_names_by_layer.values()
            for name in layer_names
        }
        source_physical_names = [
            name for name in physical_names
            if name not in sidecar_scale_set and name not in nf12_name_set
        ]

        # F128: a deterministic raw-safetensors fast tier (see
        # formats/kimi_k3_fast_tier.py) -- distinct from the vpack2 overlay
        # above, for checkpoints that were never packed at all. Partition
        # BEFORE grouping by shard so the slow-tier by_shard loop below
        # never even opens a shard file for a name the fast tier already
        # covers.
        self._ensure_raw_fast_tier_loaded()
        fast_names = [
            n for n in source_physical_names
            if (self.raw_fast_tier_enabled
                and n in self._raw_fast_tier_manifest
                and n not in self._mtp_proposal_plain_names
                and (
                    n not in self._mtplx_mtp_sidecar_names
                    or n in self._mtplx_mtp_exact_fast_names))]
        fast_name_set = set(fast_names)
        slow_names = [
            n for n in source_physical_names
            if n not in fast_name_set]

        by_shard: dict[str, list[str]] = defaultdict(list)
        for n in slow_names:
            by_shard[self.weight_map[n]].append(n)
        if self.safetensors_offset_order:
            for shard, shard_names in by_shard.items():
                shard_names.sort(
                    key=lambda name: self._safetensors_physical_offset(
                        shard, name
                    )
                )

        # mx.load() only creates lazy file-backed arrays. The SMB read that can
        # fail happens in mx.eval(), so retry the complete load+select+eval
        # transaction, not just the cheap metadata/open operation.
        t0 = time.perf_counter()
        for attempt in range(4):
            out: dict[str, mx.array] = {}
            sidecar_pool = None
            sidecar_future = None
            try:
                if sidecar_scale_names:
                    # The encoded scale stream lives in a distinct immutable
                    # file. Read and reconstruct it on one worker while the
                    # main thread faults the much larger released FP4 payloads.
                    # Routing is already authoritative and the same exact
                    # scales are joined before QTensor construction; this is
                    # scheduling overlap, not speculative data access.
                    import concurrent.futures as cf

                    sidecar_pool = cf.ThreadPoolExecutor(max_workers=1)
                    sidecar_future = sidecar_pool.submit(
                        self._decode_k3_scale_sidecars,
                        sidecar_scale_names,
                    )
                raw_parallel_tiers = bool(
                    self.parallel_storage_reads
                    and fast_names
                    and by_shard
                    and self._raw_fast_tier_is_independent(fast_names)
                )

                def load_slow_names() -> tuple[dict[str, mx.array], int]:
                    slow_out: dict[str, mx.array] = {}
                    for shard, shard_names in by_shard.items():
                        lazy = self._load_shard(self.dir / shard)
                        for name in shard_names:
                            slow_out[name] = lazy[
                                self._real_name.get(name, name)
                            ]
                    mx.eval(list(slow_out.values()))
                    return (
                        slow_out,
                        sum(int(array.nbytes) for array in slow_out.values()),
                    )

                if raw_parallel_tiers:
                    # The worker handles the independently-mounted raw overlay
                    # while this thread faults the checkpoint arrays.  Device
                    # identity and the explicit scheduling gate were both
                    # checked above; same-device and A/B-control paths remain
                    # serial.
                    fast_executor = (
                        self._raw_fast_tier_executor_for_reads()
                    )
                    parallel_started_ns = time.perf_counter_ns()

                    def load_fast_names_timed():
                        started_ns = time.perf_counter_ns()
                        value = self._read_raw_fast_tier_tensors(fast_names)
                        return value, time.perf_counter_ns() - started_ns

                    fast_future = fast_executor.submit(
                        load_fast_names_timed)
                    archive_started_ns = time.perf_counter_ns()
                    slow_out, slow_bytes = load_slow_names()
                    archive_service_ns = (
                        time.perf_counter_ns() - archive_started_ns)
                    (fast_out, fast_bytes), fast_service_ns = (
                        fast_future.result())
                    parallel_wall_ns = (
                        time.perf_counter_ns() - parallel_started_ns)
                    out.update(slow_out)
                    out.update(fast_out)
                    self.fast_tier_bytes += fast_bytes
                    self.fast_tier_tensors += len(fast_names)
                    self.archive_bytes += slow_bytes
                    self._record_parallel_tier(
                        fast_bytes=fast_bytes,
                        archive_bytes=slow_bytes,
                        wall_ns=parallel_wall_ns,
                        fast_service_ns=fast_service_ns,
                        archive_service_ns=archive_service_ns,
                    )
                elif fast_names:
                    fast_out, fast_bytes = (
                        self._read_raw_fast_tier_tensors(fast_names)
                    )
                    out.update(fast_out)
                    self.fast_tier_bytes += fast_bytes
                    self.fast_tier_tensors += len(fast_names)
                    if by_shard:
                        slow_out, slow_bytes = load_slow_names()
                        out.update(slow_out)
                        self.archive_bytes += slow_bytes
                elif by_shard:
                    slow_out, slow_bytes = load_slow_names()
                    out.update(slow_out)
                    self.archive_bytes += slow_bytes
                nbytes = sum(a.nbytes for a in out.values())
                for layer, nf12_names in nf12_names_by_layer.items():
                    nf12_out, nf12_bytes = self._decode_bf16_nf12_layer(
                        layer, nf12_names
                    )
                    out.update(nf12_out)
                    nbytes += nf12_bytes
                if sidecar_future is not None:
                    sidecar_out, sidecar_bytes = sidecar_future.result()
                    out.update(sidecar_out)
                    nbytes += sidecar_bytes
                    sidecar_pool.shutdown(wait=True)
                    sidecar_pool = None
                if self._dsv4_aux or self._glm53_fp8_aux:
                    out = self._join_released_block_fp8(names, out)
                elif self._quant_aux or self._ct_int4_aux or self._ct_mxfp4_aux:
                    from .quant import (
                        QTensor, dequantize_compressed_tensors_int4,
                        dequantize_compressed_tensors_mxfp4)

                    logical: dict = {}
                    for name in names:
                        ct_aux = self._ct_int4_aux.get(name)
                        if ct_aux is not None:
                            shape = tuple(int(v) for v in out[ct_aux.shape].tolist())
                            dequant = dequantize_compressed_tensors_int4(
                                out[ct_aux.packed], out[ct_aux.scale], shape)
                            # 2026-07-19: MLX >=0.31.2 binds ops to their
                            # creation thread's stream (streams are now
                            # thread-local). dequantize_compressed_tensors_int4
                            # returns a LAZY graph (arange/shift/mask/reshape/
                            # cast, never eval'd internally) -- unlike the raw
                            # `out` tensors just above, which were eval'd on
                            # THIS thread at line 428. Left lazy, this graph is
                            # first materialized wherever the caller (the main
                            # thread, via matmul) eventually evals it -- but
                            # when this fetch runs on the prefetch thread
                            # (K2.5's compressed-tensors INT4 experts are the
                            # only checkpoint format that takes this branch),
                            # that is a DIFFERENT thread than the one the ops
                            # were constructed on, and its stream is not
                            # registered there: "RuntimeError: There is no
                            # Stream(gpu, N) in current thread." Force eval
                            # here, on the thread that built the graph.
                            mx.eval(dequant)
                            logical[name] = dequant
                            continue
                        ct_mxfp4_aux = self._ct_mxfp4_aux.get(name)
                        if ct_mxfp4_aux is not None:
                            # F128: no .weight_shape tensor exists for K3's
                            # MXFP4 pairs (confirmed on the real checkpoint);
                            # logical shape is inferred, safe because MXFP4
                            # always packs exactly 2 FP4 values per byte.
                            packed_arr = out[ct_mxfp4_aux.packed]
                            shape = (packed_arr.shape[0], packed_arr.shape[1] * 2)
                            scale_arr = out[ct_mxfp4_aux.scale]
                            transform_t0 = time.perf_counter_ns()
                            if self.native_ct_mxfp4:
                                if (
                                    packed_arr.dtype != mx.uint8
                                    or scale_arr.dtype != mx.uint8
                                    or packed_arr.shape[1] % 4
                                ):
                                    raise ValueError(
                                        "native compressed-tensors MXFP4 needs "
                                        "uint8 packed/scale tensors and a packed "
                                        "row width divisible by four bytes")
                                # MLX packs eight E2M1 nibbles per uint32 lane,
                                # low-to-high bits. compressed-tensors stores the
                                # identical nibble stream as four adjacent uint8
                                # bytes; this view changes neither bytes nor order.
                                qtensor = QTensor(
                                    packed_arr.reshape(
                                        packed_arr.shape[0], -1).view(mx.uint32),
                                    scale_arr, None, 4, 32, "mxfp4")
                                # A fetch may execute on the prefetch thread.
                                # Materialize the view there so no lazy graph is
                                # first evaluated on another thread's MLX stream.
                                mx.eval(qtensor.wq, qtensor.scales)
                                logical[name] = qtensor
                                resident_bytes = qtensor.nbytes
                            else:
                                dequant = dequantize_compressed_tensors_mxfp4(
                                    packed_arr, scale_arr, shape)
                                # Same thread-stream reasoning as the INT4 branch
                                # above applies identically here.
                                mx.eval(dequant)
                                logical[name] = dequant
                                resident_bytes = dequant.nbytes
                            self._record_ct_mxfp4_transform(
                                elapsed_ns=(
                                    time.perf_counter_ns() - transform_t0),
                                input_bytes=(
                                    packed_arr.nbytes + scale_arr.nbytes),
                                resident_bytes=resident_bytes,
                            )
                            continue
                        aux = self._quant_aux.get(name)
                        if aux is None:
                            logical[name] = out[name]
                            continue
                        logical[name] = QTensor(
                            out[name], out[aux.scales],
                            out[aux.biases] if aux.biases is not None else None,
                            aux.bits, aux.group_size, aux.mode,
                        )
                    out = logical
                return out, time.perf_counter() - t0, nbytes
            except (OSError, RuntimeError):
                if sidecar_pool is not None:
                    sidecar_pool.shutdown(wait=True, cancel_futures=True)
                # Discard every partially materialized/lazy array before retry;
                # otherwise stale file descriptors and half-read allocations can
                # survive into the next attempt.
                out.clear()
                mx.clear_cache()
                if attempt == 3:
                    raise
                self._recover_nas_mount()
                time.sleep(5 * (2 ** attempt))
            except BaseException:
                if sidecar_pool is not None:
                    sidecar_pool.shutdown(wait=True, cancel_futures=True)
                raise

        raise AssertionError("unreachable raw fetch retry state")

    def _recover_nas_mount(self) -> None:
        """Remount/re-resolve this model after a transient storage failure."""
        storage = get_storage_config()
        if not storage.is_configured_path(self.dir):
            return
        candidate = storage.resolve(self.dir)
        if candidate != self.dir:
            print(f"[store] re-resolved model dir -> {candidate}", flush=True)
            self.dir = candidate
            self.vpack = candidate / "weights.vpack"

    def _fetch_packed(self, names: list[str]) -> tuple[dict[str, mx.array], float, int]:
        import sys

        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        from formats.packed import read_tensor_bytes, to_mx

        t0 = time.perf_counter()
        out: dict = {}
        nbytes = 0
        eval_list = []
        by_file: dict[str, list[str]] = defaultdict(list)
        for n in names:
            by_file[self.weight_map[n]].append(n)
        for fname, fnames in by_file.items():
            root = next((d for d in self.fast_dirs if (d / fname).exists()), self.vpack)
            if fname.endswith(".safetensors"):
                # SQ q4 stores: pre-quantized expert triplets (-> QTensor) and
                # bf16 remainder chunks live in plain safetensors files.
                lazy = self._load_shard(root / fname)
                nbytes += (root / fname).stat().st_size
                for n in fnames:
                    if f"{n}.wq" in lazy:
                        from .quant import QTensor

                        q = QTensor(lazy[f"{n}.wq"], lazy[f"{n}.scales"],
                                    lazy[f"{n}.biases"], 4, 64)
                        out[n] = q
                        eval_list += [q.wq, q.scales, q.biases]
                    else:
                        out[n] = lazy[n]
                        eval_list.append(lazy[n])
                continue
            for n in fnames:
                nbytes += (root / self.weight_map[n]).stat().st_size
                head, raw = read_tensor_bytes(root, self.weight_map[n])
                out[n] = to_mx(head, raw)
                eval_list.append(out[n])
        mx.eval(eval_list)
        return out, time.perf_counter() - t0, nbytes

    def _load_shard(self, path: Path, attempts: int = 4):
        """Tier-3 (network storage) resilience: SMB shares drop mid-run — a
        44-minute GLM sweep died to exactly this. Retry with backoff; if the
        volume itself vanished, attempt to remount the configured share."""
        import os
        import time as _t

        for i in range(attempts):
            try:
                return mx.load(str(path))
            except RuntimeError:
                if i == attempts - 1:
                    raise
                if not path.exists() and get_storage_config().is_configured_path(path):
                    self._recover_nas_mount()
                    _t.sleep(3)
                    path = self.dir / path.name
                _t.sleep(5 * (2 ** i))

    def fetch_layer(self, layer: int) -> tuple[dict[str, mx.array], float, int]:
        return self.fetch(self.layer_param_names(layer))
