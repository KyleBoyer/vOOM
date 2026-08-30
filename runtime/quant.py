"""Quantize-on-load: weights are stored on disk at full precision (bf16/fp16) and
optionally quantized as they enter the WeightCache. Disk reads stay full-precision;
the *resident* footprint shrinks 4-8x, which lets far more (often all) layers stay
cached. This trades quantization error for residency — configurable per module so
attention can stay bf16 while the MLP goes 4-bit.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class QTensor:
    """A weight quantized with mx.quantize. Behaves as a matmul-able weight via
    layer_runner, which dispatches to mx.quantized_matmul."""

    wq: mx.array
    scales: mx.array
    biases: mx.array | None
    bits: int
    group_size: int
    mode: str = "affine"

    @property
    def nbytes(self) -> int:
        return (
            self.wq.nbytes + self.scales.nbytes
            + (self.biases.nbytes if self.biases is not None else 0)
        )

    @property
    def shape(self) -> tuple[int, ...]:
        # MLX packs 32 / bits logical columns into each uint32 lane.
        # MLX may pack widths that do not divide one uint32 lane (for example
        # affine 3-bit). Preserve the remainder until after multiplication:
        # six packed lanes represent 64 logical 3-bit columns, not 60.
        return (*self.wq.shape[:-1], self.wq.shape[-1] * 32 // self.bits)

    @property
    def dtype(self):
        # MX/NV floating-point modes store encoded scales as uint8; that is not
        # the logical dequantized dtype. MLX defaults those modes to bfloat16.
        return (self.scales.dtype
                if mx.issubdtype(self.scales.dtype, mx.floating)
                else mx.bfloat16)


@dataclass
class RerankedQHead:
    """Approximate full-vocabulary head with exact BF16 candidate scoring.

    The quantized projection cheaply finds a small candidate set.  Only those
    rows are then multiplied from the original BF16 head with ``gather_mm``;
    non-candidates are masked. The resulting sparse logits retain the exact
    candidate winner for greedy decode without paying for a full BF16 vocabulary
    projection every token; categorical sampling is correspondingly restricted
    to the empirically selected candidate support in this explicitly lossy mode.

    Candidate recall is an empirical property, not a mathematical guarantee.
    This representation is therefore restricted to an explicitly lossy
    profile. ``exact`` may be either the historical resident BF16 matrix or a
    verified row provider that reads only selected BF16 rows.
    """

    exact: object
    approx: QTensor
    candidates: int
    recall_probe_every: int = 0
    calls: int = 0
    positions: int = 0
    candidate_winner_changes: int = 0
    candidate_recall_probes: int = 0
    candidate_recall_hits: int = 0
    # Optional explicit, ranks-only evidence writer. It is attached only for
    # a fingerprint-bound row-paged K=64 capture run. The scope defaults to
    # authoritative target decisions; Qwen's shared-head MTP drafter marks
    # its proposal projections separately so they can never enter the gate.
    recall_rank_capture: object | None = None
    recall_rank_capture_scope: str = "authoritative-target"

    @property
    def nbytes(self) -> int:
        return int(getattr(self.exact, "nbytes", 0) or 0) + self.approx.nbytes

    @property
    def shape(self) -> tuple[int, ...]:
        shape = getattr(self.exact, "shape", None)
        if shape is not None:
            return tuple(shape)
        return (int(self.exact.vocab), int(self.exact.hidden))

    @property
    def dtype(self):
        dtype = getattr(self.exact, "dtype", mx.bfloat16)
        return mx.bfloat16 if dtype == "BF16" else dtype

    def record_candidate_recall(self, hits: int, probes: int) -> None:
        if hits < 0 or probes < 0 or hits > probes:
            raise ValueError("candidate recall requires 0 <= hits <= probes")
        self.candidate_recall_hits += int(hits)
        self.candidate_recall_probes += int(probes)

    def telemetry_snapshot(self) -> dict[str, int]:
        values = {
            "calls": int(self.calls),
            "positions": int(self.positions),
            "candidate_winner_changes": int(self.candidate_winner_changes),
            "candidate_recall_probes": int(self.candidate_recall_probes),
            "candidate_recall_hits": int(self.candidate_recall_hits),
        }
        provider = getattr(self.exact, "candidate_telemetry", None)
        if callable(provider):
            values.update(provider())
        capture_snapshot = getattr(
            self.recall_rank_capture, "telemetry_snapshot", None)
        if callable(capture_snapshot):
            values.update({
                f"candidate_rank_capture_{key}": int(value)
                for key, value in capture_snapshot().items()
            })
        return values


@contextmanager
def reranked_lm_head_capture_scope(head, scope: str):
    """Temporarily label one shared-head projection for recall evidence.

    The ordinary target path remains ``authoritative-target``. Proposal-only
    MTP projections use ``mtp-draft`` and are excluded by ``reranked_matmul``.
    """

    if not isinstance(head, RerankedQHead):
        yield
        return
    previous = head.recall_rank_capture_scope
    head.recall_rank_capture_scope = str(scope)
    try:
        yield
    finally:
        head.recall_rank_capture_scope = previous


def make_reranked_q_head(
    exact: mx.array,
    *,
    candidates: int = 32,
    group_size: int = 32,
    bits: int = 4,
    mode: str = "mxfp4",
) -> RerankedQHead:
    """Build and materialize the approximate half of a reranked LM head."""
    if exact.ndim != 2:
        raise ValueError("reranked LM head must be a rank-2 matrix")
    if candidates <= 0 or candidates > exact.shape[0]:
        raise ValueError(
            f"rerank candidates must be in [1, {exact.shape[0]}], got {candidates}"
        )
    policy = QuantPolicy(bits=bits, group_size=group_size, mode=mode, min_dim=0)
    if exact.shape[1] % group_size:
        raise ValueError(
            f"LM head width {exact.shape[1]} is not divisible by group_size={group_size}"
        )
    packed = mx.quantize(
        exact, group_size=group_size, bits=bits, mode=mode
    )
    mx.eval(packed)
    approx = QTensor(
        packed[0], packed[1], packed[2] if len(packed) > 2 else None,
        policy.bits, policy.group_size, policy.mode,
    )
    return RerankedQHead(exact=exact, approx=approx, candidates=candidates)


def make_row_paged_reranked_q_head(
    approx: QTensor,
    exact_rows,
    *,
    candidates: int = 64,
    recall_probe_every: int = 0,
) -> RerankedQHead:
    """Compose an on-disk quantized shortlist with a BF16 row provider."""

    shape = tuple(getattr(exact_rows, "shape", (
        getattr(exact_rows, "vocab", 0), getattr(exact_rows, "hidden", 0))))
    if len(shape) != 2 or shape != approx.shape:
        raise ValueError(
            f"approximate/exact LM-head shapes differ: {approx.shape} vs {shape}")
    if candidates <= 0 or candidates > shape[0]:
        raise ValueError(
            f"rerank candidates must be in [1, {shape[0]}], got {candidates}")
    if recall_probe_every < 0:
        raise ValueError("recall_probe_every must be non-negative")
    if not callable(getattr(exact_rows, "candidate_logits", None)):
        raise ValueError("exact LM-head row provider lacks candidate_logits")
    return RerankedQHead(
        exact=exact_rows, approx=approx, candidates=candidates,
        recall_probe_every=recall_probe_every)


@dataclass
class QuantPolicy:
    bits: int = 4
    group_size: int = 64
    mode: str = "affine"
    quantize_attention: bool = True
    quantize_mlp: bool = True
    quantize_router: bool = True
    quantize_lm_head: bool = True
    min_dim: int = 512  # leave small projections alone

    def __post_init__(self):
        valid = (
            self.mode == "affine"
            and self.group_size in (32, 64, 128)
            and self.bits in (2, 3, 4, 5, 6, 8)
        ) or (
            (self.mode, self.group_size, self.bits)
            in {("mxfp4", 32, 4), ("mxfp8", 32, 8), ("nvfp4", 16, 4)}
        )
        if not valid:
            raise ValueError(
                f"unsupported MLX quantization parameters: mode={self.mode!r}, "
                f"group_size={self.group_size}, bits={self.bits}"
            )

    def wants(self, name: str, arr: mx.array) -> bool:
        if isinstance(arr, QTensor):
            return False
        if arr.ndim != 2 or not name.endswith(".weight"):
            return False
        if min(arr.shape) < self.min_dim or arr.shape[1] % self.group_size:
            return False
        if "embed_tokens" in name or "norm" in name:
            return False
        if ".self_attn." in name:
            return self.quantize_attention
        # 2026-07-19 (benchmark-sweep follow-up): Kimi's MoE module is named
        # "block_sparse_moe", not "mlp" -- without this OR, NONE of its
        # expert weights (the dominant byte mass across 26 of 27 layers)
        # ever matched ".mlp." below, so "lossy" mode silently left them at
        # full bf16 precision. Measured effect: Kimi-Linear-48B-A3B-Instruct
        # showed IDENTICAL tok/s in lossless vs lossy mode before this fix.
        # Preserves the exact pre-existing control flow/quirk below
        # (a gate weight only short-circuits to `False` when quantize_router
        # is False; otherwise it falls through and is actually governed by
        # quantize_mlp, not quantize_router -- not touched here).
        if ((name.endswith(".mlp.gate.weight")
             or name.endswith(".mlp.shared_expert_gate.weight")
             or name.endswith(".block_sparse_moe.gate.weight"))
                and not self.quantize_router):
            return False
        if ".mlp." in name or ".block_sparse_moe." in name:
            return self.quantize_mlp
        return self.quantize_lm_head and "lm_head" in name

    def transform(self, name: str, arr: mx.array):
        # A standard MLX checkpoint may already store this tensor quantized on
        # disk. Preserve that representation instead of trying to quantize its
        # packed uint32 payload a second time.
        if isinstance(arr, QTensor):
            return arr
        if not self.wants(name, arr):
            return arr
        packed = mx.quantize(
            arr, group_size=self.group_size, bits=self.bits, mode=self.mode)
        mx.eval(packed)
        return QTensor(
            packed[0], packed[1], packed[2] if len(packed) > 2 else None,
            self.bits, self.group_size, self.mode)


_DEQUANT_INT4_SOURCE = """
    uint row = thread_position_in_grid.y;
    uint col = thread_position_in_grid.x;
    if (col >= COLS) return;
    uint num_words = packed_shape[1];
    uint num_groups = scale_shape[1];
    uint word_idx = col / 8;
    uint nibble_shift = (col % 8) * 4;
    uint word = packed[row * num_words + word_idx];
    uint nibble = (word >> nibble_shift) & 0xFu;
    int signed_val = int(nibble) - 8;
    uint group_idx = col / GROUP_SIZE;
    float s = float(scale[row * num_groups + group_idx]);
    out[row * COLS + col] = T(float(signed_val) * s);
"""

_dequant_int4_kernel_cache: dict[tuple[int, int], "mx.fast.metal_kernel"] = {}


def _dequant_int4_kernel(cols: int, group_size: int):
    """F107 (2026-07-25): cols/group_size baked in as source-literal constants
    (not runtime shape lookups) so the compiled kernel needs no per-call
    division -- cached per (cols, group_size) pair since a real checkpoint
    only has a handful of distinct expert-weight shapes (gate/up/down_proj),
    so this compiles at most a few times per process, not once per call."""
    key = (cols, group_size)
    kernel = _dequant_int4_kernel_cache.get(key)
    if kernel is None:
        source = _DEQUANT_INT4_SOURCE.replace(
            "COLS", str(cols)).replace("GROUP_SIZE", str(group_size))
        kernel = mx.fast.metal_kernel(
            name=f"dequant_ct_int4_{cols}_{group_size}",
            input_names=["packed", "scale"],
            output_names=["out"],
            source=source,
        )
        _dequant_int4_kernel_cache[key] = kernel
    return kernel


def dequantize_compressed_tensors_int4(
    packed: mx.array, scale: mx.array, shape: tuple[int, int], packed_dim: int = 1,
) -> mx.array:
    """Dequantize a vllm-project/compressed-tensors "pack-quantized" INT4 weight.

    F93 (docs/future_lossless_techniques.md): this is Kimi K2.5's AS-RELEASED
    expert-weight format on Hugging Face (`.weight_packed`/`.weight_scale`/
    `.weight_shape` tensor triples in place of an ordinary `.weight`) --
    NOT the same scheme as this project's own QTensor/mx.quantize (different
    library, different bit layout/scale convention; do not conflate them).
    Only the MoE expert FFN weights use this format in K2.5's checkpoint;
    attention and router weights are ordinary bf16 `.weight` tensors.

    Algorithm verified bit-exact (2026-07-18) against the real
    `compressed_tensors.compressors.pack_quantized.helpers.unpack_from_int32`
    source (num_bits=4, which divides 32 evenly, so no value ever crosses an
    int32 word boundary -- the general cross-word-overflow case in the real
    function is dead code for this specific bit width and is not
    reimplemented here): 8 signed int4 values per int32 word, value i at bit
    offset `i*4`, stored with a +8 offset (i.e. raw nibble `0..15` encodes
    signed `-8..7`); symmetric groupwise scale (no zero-point tensor in this
    checkpoint), one BF16 scale per `group_size` consecutive elements along
    `packed_dim`.

    F107 (2026-07-25): a real per-op profile of the original 5-op MLX
    composite (bit-shift+mask, reshape, signed-offset, scale mx.repeat,
    final multiply+cast) against a real K2.5 expert-weight shape (2048,
    7168) found NO single dominant op -- each cost 1-4ms, ~9.2ms total --
    and, critically, that this dequantization is ~97.6% of the COMBINED
    dequant+matmul time for one expert (9.2ms dequant vs 0.5ms matmul),
    unlike this session's RMSNorm findings where the op was negligible next
    to its surrounding matmul. This single `mx.fast.metal_kernel` fuses all
    5 steps into one dispatch (no intermediate nibbles/signed/scale-expanded
    tensors ever materialized) -- verified byte-identical (0.0 max abs diff)
    against the original composite, isolated speedup 7.48x (9.298ms ->
    1.243ms) at this same real shape. See docs/future_lossless_techniques.md
    F107 for the real end-to-end verdict this produced against the real
    K2.5 checkpoint -- not trusted from this isolated number alone.

    :param packed: int32 tensor, the `.weight_packed` tensor as loaded
    :param scale: BF16 tensor, the `.weight_scale` tensor as loaded --
        shape (rows, cols // group_size) for packed_dim=1
    :param shape: the true logical (pre-pack) shape, from `.weight_shape`
    :param packed_dim: which logical axis was packed (0 or 1); K2.5 uses 1
    :returns: dequantized weight, shape `shape`, dtype matching `scale`
    """
    if packed_dim != 1:
        raise NotImplementedError("F93: only packed_dim=1 verified/needed for K2.5 so far")
    rows, cols = shape
    if packed.shape != (rows, -(-cols // 8)):
        raise ValueError(
            f"packed shape {packed.shape} inconsistent with logical shape {shape} "
            "for 8-values-per-int32 (num_bits=4) packing")
    if cols % scale.shape[1]:
        raise ValueError(
            f"weight_scale shape {scale.shape} does not evenly divide "
            f"logical cols {cols} (implied group_size {cols / scale.shape[1]})")
    group_size = cols // scale.shape[1]
    kernel = _dequant_int4_kernel(cols, group_size)
    out = kernel(
        inputs=[packed.astype(mx.uint32), scale],
        template=[("T", scale.dtype)],
        grid=(cols, rows, 1),
        threadgroup=(min(cols, 256), 1, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[scale.dtype],
    )[0]
    return out


_MXFP4_E2M1_MAGNITUDE_LUT = mx.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=mx.float32)


def dequantize_compressed_tensors_mxfp4(
    packed: mx.array, scale: mx.array, shape: tuple[int, int],
    packed_dim: int = 1, out_dtype=mx.bfloat16,
) -> mx.array:
    """Dequantize a vllm-project/compressed-tensors "mxfp4-pack-quantized"
    weight -- Kimi K3's real, as-released expert-weight format on Hugging
    Face (`moonshotai/Kimi-K3`, checked directly from its real config.json
    `quantization_config` block: ``format: "mxfp4-pack-quantized"``,
    ``quant_method: "compressed-tensors"``, ``group_size: 32``,
    ``num_bits: 4``, ``symmetric: true``, ``scale_dtype: torch.uint8``, and
    ``ignore: ["re:.*self_attn.*", ...]`` confirming only MoE expert FFN
    weights use this format -- attention stays ordinary BF16, matching
    K2.5's own "experts only" pattern). This is a DIFFERENT numeric format
    from K2.5's `dequantize_compressed_tensors_int4` (integer, group_size
    128 there) despite both being "compressed-tensors" family: MXFP4 packs
    FP4 E2M1 codes, not signed integers, at group_size=32.

    Algorithm verified 2026-07-27 against the real, installed
    `compressed-tensors==0.17.1` library source (pip-installed for this
    verification only, not a core runtime dependency), specifically:
    - `compressed_tensors.compressors.nvfp4.helpers.unpack_fp4_from_uint8`
      (MXFP4PackedCompressor inherits NVFP4's packing/unpacking unchanged --
      only scale compression differs) -- 2 E2M1 4-bit codes per uint8 byte,
      low nibble first: bit 3 (0x08) is sign, bits 0-2 (0x07) index a fixed
      8-entry magnitude lookup table `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
      6.0]` (this IS the OCP E2M1 format's exact representable value set,
      not something K3-specific).
    - `compressed_tensors.compressors.mx_utils.decompress_mx_scale` -- each
      uint8 scale byte is a biased E8M0 exponent: `2 ** (byte - 127)`.
    - `compressed_tensors.quantization.lifecycle.forward_helpers._dequantize`
      -- final value = unpacked_e2m1 * scale (no zero_point, since
      symmetric=true omits that tensor entirely; no division by a
      `global_scale`, confirmed EMPIRICALLY by inspecting a real downloaded
      K3 shard directly -- no `.weight_global_scale` tensor exists
      alongside `.weight_packed`/`.weight_scale` for any real expert layer
      checked, even though NVFP4PackedCompressor's generic
      `compression_param_names` classmethod always lists it; Moonshot's own
      export omits it for this single-level MXFP4 scaling, unlike NVFP4's
      real two-level design).

    Real shapes confirmed from `moonshotai/Kimi-K3`'s actual downloaded
    checkpoint (layer 4, expert 0, w1): `weight_packed` (3072, 1792) uint8,
    `weight_scale` (3072, 112) uint8 -- 1792*2=3584 logical columns,
    3584/112=32 group_size, matching config exactly.

    :param packed: uint8 tensor, the `.weight_packed` tensor as loaded,
        shape (rows, cols // 2) for packed_dim=1 (2 FP4 values/byte)
    :param scale: uint8 tensor, the `.weight_scale` tensor as loaded (E8M0
        biased exponents), shape (rows, cols // group_size)
    :param shape: the true logical (pre-pack) shape, from `.weight_shape`
        if present, or the checkpoint's own declared logical shape
    :param packed_dim: which logical axis was packed; K3's real experts use 1
    :param out_dtype: output float dtype (K3's own config.json declares its
        overall model dtype as bfloat16, matching this default)
    :returns: dequantized weight, shape `shape`, dtype `out_dtype`
    """
    if packed_dim != 1:
        raise NotImplementedError(
            "F128: only packed_dim=1 verified/needed for Kimi K3 so far")
    rows, cols = shape
    if packed.shape != (rows, cols // 2) or cols % 2:
        raise ValueError(
            f"packed shape {packed.shape} inconsistent with logical shape "
            f"{shape} for 2-values-per-uint8 (E2M1 FP4) packing")
    if cols % scale.shape[1]:
        raise ValueError(
            f"weight_scale shape {scale.shape} does not evenly divide "
            f"logical cols {cols} (implied group_size {cols / scale.shape[1]})")
    group_size = cols // scale.shape[1]

    packed_u8 = packed.astype(mx.uint8)
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    # Interleave low/high back into logical column order: byte j holds
    # logical columns 2j (low nibble) and 2j+1 (high nibble), matching the
    # real unpack_fp4_from_uint8's stack-then-flatten order exactly.
    nibbles = mx.stack([low, high], axis=-1).reshape(rows, cols)

    sign = (nibbles & 0x08).astype(mx.bool_)
    magnitude_index = (nibbles & 0x07).astype(mx.uint32)
    magnitude = _MXFP4_E2M1_MAGNITUDE_LUT[magnitude_index]
    values = mx.where(sign, -magnitude, magnitude)

    scale_exp = scale.astype(mx.int32) - 127
    scale_float = 2.0 ** scale_exp.astype(mx.float32)
    scale_expanded = mx.repeat(scale_float, group_size, axis=1)

    return (values * scale_expanded).astype(out_dtype)


def reranked_matmul(
    x: mx.array,
    w: RerankedQHead,
    logits_transform=None,
) -> mx.array:
    """Candidate search plus exact row scoring, optionally after a mask.

    Applying a grammar before candidate selection is essential: an
    unrestricted top-k can otherwise contain no legal token, at which point
    masking the already-sparse result produces an all--infinity distribution.
    """
    approx = matmul(x, w.approx)
    selection = approx
    if logits_transform is not None:
        selection = logits_transform(approx.reshape(-1)).reshape(approx.shape)
    k = w.candidates
    indices = mx.argpartition(
        -selection, kth=k - 1, axis=-1)[..., :k]

    provider = getattr(w.exact, "candidate_logits", None)
    if callable(provider):
        exact_scores = provider(x, indices)
    else:
        # Treat each vocabulary row as a one-output expert. gather_mm uses the
        # same matrix kernel as an exact projection for just the dynamic rows,
        # unlike an elementwise multiply+sum whose reduction arithmetic was
        # measured to change greedy choices on the real OLMoE checkpoint.
        flat = x.reshape(-1, x.shape[-1])
        flat_indices = indices.reshape(-1, k)
        lhs = mx.expand_dims(flat, (-2, -3))
        rhs = mx.expand_dims(w.exact, -2).swapaxes(-1, -2)
        exact_scores = mx.gather_mm(
            lhs, rhs, rhs_indices=flat_indices
        ).squeeze((-1, -2)).reshape(indices.shape)

    call_number = w.calls + 1
    probe_due = bool(
        callable(provider) and w.recall_probe_every
        and call_number % w.recall_probe_every == 0)
    capture = w.recall_rank_capture
    capture_remaining = int(getattr(capture, "remaining", 0) or 0)
    capture_due = bool(
        callable(provider)
        and w.recall_rank_capture_scope == "authoritative-target"
        and capture_remaining > 0)
    if probe_due or capture_due:
        oracle = getattr(w.exact, "candidate_recall_logits", None)
        if not callable(oracle):
            raise ValueError(
                "candidate-recall probing requires a full-logit row provider")
        flat_x = x.reshape(-1, x.shape[-1])
        total_positions = int(flat_x.shape[0])
        oracle_positions = (
            total_positions if probe_due
            else min(total_positions, capture_remaining))
        oracle_x = (
            x if oracle_positions == total_positions
            else flat_x[:oracle_positions])
        exact_full = oracle(oracle_x)
        if logits_transform is not None:
            exact_full = logits_transform(
                exact_full.reshape(-1)).reshape(exact_full.shape)
        exact_rows = exact_full.reshape(-1, exact_full.shape[-1])
        exact_full_winner = mx.argmax(exact_rows, axis=-1, keepdims=True)
        selected_indices = indices.reshape(-1, k)[:oracle_positions]
        recall_hit_rows = mx.any(
            selected_indices == exact_full_winner, axis=-1)
        if probe_due:
            recall_hits = mx.sum(recall_hit_rows)
            mx.eval(recall_hits)
            w.record_candidate_recall(
                int(recall_hits.item()), int(exact_full_winner.size))
        if capture_due:
            selection_rows = selection.reshape(-1, selection.shape[-1])[
                :oracle_positions]
            winner_values = mx.take_along_axis(
                selection_rows, exact_full_winner, axis=-1)
            vocabulary_ids = mx.arange(selection_rows.shape[-1]).reshape(1, -1)
            greater = mx.sum(selection_rows > winner_values, axis=-1)
            tied_lower_ids = mx.sum(
                (selection_rows == winner_values)
                & (vocabulary_ids < exact_full_winner),
                axis=-1,
            )
            stable_ranks = 1 + greater + tied_lower_ids
            top1_agreements = (
                mx.argmax(selection_rows, axis=-1)
                == exact_full_winner.reshape(-1))
            mx.eval(stable_ranks, recall_hit_rows, top1_agreements)
            capture.record(
                stable_ranks.tolist(),
                recall_hit_rows.tolist(),
                top1_agreements.tolist(),
            )

    w.calls = call_number
    w.positions += int(indices.size // k)
    if callable(provider):
        # The row-paged path already synchronizes candidate IDs and scores for
        # I/O. Reuse that synchronization to disclose how often exact scoring
        # changes the approximate top-1 inside the shortlist. This is not
        # mislabeled as full-vocabulary recall; oracle recall has separate
        # hit/probe counters populated by candidate-recall fixtures.
        mx.eval(selection, indices, exact_scores)
        approximate_winner = mx.argmax(selection, axis=-1)
        exact_choice = mx.argmax(exact_scores, axis=-1, keepdims=True)
        exact_winner = mx.take_along_axis(
            indices, exact_choice, axis=-1).squeeze(-1)
        mx.eval(approximate_winner, exact_winner)
        w.candidate_winner_changes += int(np.sum(
            np.asarray(approximate_winner) != np.asarray(exact_winner)))

    sparse = mx.full(
        approx.shape, float("-inf"), dtype=approx.dtype)
    result = mx.put_along_axis(
        sparse, indices, exact_scores.astype(approx.dtype), axis=-1)
    if logits_transform is not None:
        result = logits_transform(result.reshape(-1)).reshape(result.shape)
    return result


def matmul(x: mx.array, w) -> mx.array:
    """x @ w.T for a plain, quantized, or candidate-reranked weight."""
    from .bf16_nf12_linear import NF12Tensor

    if isinstance(w, NF12Tensor):
        return w.matmul(x)
    if isinstance(w, RerankedQHead):
        return reranked_matmul(x, w)
    if isinstance(w, QTensor):
        return mx.quantized_matmul(
            x, w.wq, scales=w.scales, biases=w.biases,
            transpose=True, group_size=w.group_size, bits=w.bits, mode=w.mode,
        )
    return x @ w.T


def _fp16_bytes_to_f32(raw: mx.array) -> mx.array:
    """raw: uint8 array whose LAST axis has size 2 (one little-endian fp16
    per pair). Same bitcast-via-uint16 trick formats/packed.py's `to_mx`
    already uses for BF16/F16 tensors elsewhere in this codebase."""
    u16 = raw[..., 0].astype(mx.uint16) | (raw[..., 1].astype(mx.uint16) << 8)
    return u16.view(mx.float16).astype(mx.float32)


def dequantize_gguf_q4_k(
    packed: mx.array, shape: tuple[int, int], out_dtype=mx.bfloat16,
) -> mx.array:
    """GGUF/ggml Q4_K ("4-bit K-quant", used for most tensors in a
    "Q4_K_M"-quantized file): super-blocks of 256 values (8 sub-blocks of
    32), each sub-block's 4-bit codes scaled/offset by a 6-bit
    (scale, min) pair, those pairs themselves scaled by one shared fp16
    (d, dmin) per super-block. Ported verbatim from the real
    ggml-org/ggml `dequantize_row_q4_K`/`get_scale_min_k4`
    (`src/ggml-quants.c`, fetched 2026-07-28) -- see
    tests/test_gguf_quant_oracle.py for the byte-for-byte verification
    against that real source (both a direct transcription and, where
    available, real downloaded GGUF tensor bytes).

    `packed`: raw uint8 bytes for this tensor, shape
    `(out_features, n_super_blocks * 144)` -- GGUF lays out one tensor as
    consecutive rows, each row as consecutive 144-byte block_q4_K structs
    covering that row's full in_features length. `shape` is the PyTorch-
    convention logical shape `(out_features, in_features)`; GGUF's own
    `ne[]`/gguf-py `.shape` reports the reverse (in_features,
    out_features) -- callers must pass the PyTorch-convention tuple here,
    already corrected.
    """
    out_features, in_features = shape
    QK_K = 256
    if in_features % QK_K:
        raise ValueError(
            f"Q4_K requires in_features ({in_features}) divisible by {QK_K}")
    n_super = in_features // QK_K
    block_bytes = 144
    if packed.shape != (out_features, n_super * block_bytes):
        raise ValueError(
            f"packed shape {packed.shape} inconsistent with logical shape "
            f"{shape} for Q4_K's {block_bytes}-byte super-blocks")

    blocks = packed.reshape(out_features, n_super, block_bytes).astype(mx.uint8)
    d = _fp16_bytes_to_f32(blocks[..., 0:2])       # (out, n_super)
    dmin = _fp16_bytes_to_f32(blocks[..., 2:4])    # (out, n_super)
    scales = blocks[..., 4:16]                     # (out, n_super, 12)
    qs = blocks[..., 16:16 + 128]                  # (out, n_super, 128)

    # get_scale_min_k4(j, scales) for j in 0..7 -- unrolled (cheap, fixed
    # 8 iterations at graph-construction time, not per-element).
    sc_list, m_list = [], []
    for j in range(8):
        if j < 4:
            sc_list.append(scales[..., j] & 63)
            m_list.append(scales[..., j + 4] & 63)
        else:
            sc_list.append((scales[..., j + 4] & 0xF) | ((scales[..., j - 4] >> 6) << 4))
            m_list.append((scales[..., j + 4] >> 4) | ((scales[..., j - 0] >> 6) << 4))
    sc = mx.stack(sc_list, axis=-1).astype(mx.float32)  # (out, n_super, 8)
    m = mx.stack(m_list, axis=-1).astype(mx.float32)    # (out, n_super, 8)

    # qs: 128 bytes -> 4 chunks of 32 bytes; chunk k's low nibbles use
    # sc[2k]/m[2k], high nibbles use sc[2k+1]/m[2k+1] -- exactly
    # dequantize_row_q4_K's four `for (n = 0; n < QK_K; n += 128)`-outer /
    # `is += 2`-per-iteration steps, unrolled.
    qs_r = qs.reshape(out_features, n_super, 4, 32)
    low = (qs_r & 0xF).astype(mx.float32)
    high = (qs_r >> 4).astype(mx.float32)
    sc_pairs = sc.reshape(out_features, n_super, 4, 2)
    m_pairs = m.reshape(out_features, n_super, 4, 2)
    d_b = d[..., None, None]
    dmin_b = dmin[..., None, None]
    y_low = d_b * sc_pairs[..., 0:1] * low - dmin_b * m_pairs[..., 0:1]
    y_high = d_b * sc_pairs[..., 1:2] * high - dmin_b * m_pairs[..., 1:2]
    y_chunks = mx.concatenate([y_low, y_high], axis=-1)  # (out, n_super, 4, 64)
    return y_chunks.reshape(out_features, in_features).astype(out_dtype)


def dequantize_gguf_q6_k(
    packed: mx.array, shape: tuple[int, int], out_dtype=mx.bfloat16,
) -> mx.array:
    """GGUF/ggml Q6_K ("6-bit K-quant", used for embedding/select tensors
    in a "Q4_K_M"-quantized file): super-blocks of 256 values (16 sub-
    blocks of 16), 6-bit codes (4 low bits + 2 high bits, separately
    packed) times one signed 8-bit scale per sub-block, times one shared
    fp16 super-block scale. Ported verbatim from the real ggml-org/ggml
    `dequantize_row_q6_K` (`src/ggml-quants.c`, fetched 2026-07-28) -- see
    tests/test_gguf_quant_oracle.py.

    `packed`/`shape` conventions match `dequantize_gguf_q4_k` exactly
    (PyTorch-convention `(out_features, in_features)`, GGUF's own
    ne-order reversed).
    """
    out_features, in_features = shape
    QK_K = 256
    if in_features % QK_K:
        raise ValueError(
            f"Q6_K requires in_features ({in_features}) divisible by {QK_K}")
    n_super = in_features // QK_K
    block_bytes = 210  # ql(128) + qh(64) + scales(16) + d(2)
    if packed.shape != (out_features, n_super * block_bytes):
        raise ValueError(
            f"packed shape {packed.shape} inconsistent with logical shape "
            f"{shape} for Q6_K's {block_bytes}-byte super-blocks")

    blocks = packed.reshape(out_features, n_super, block_bytes)
    ql = blocks[..., 0:128].astype(mx.uint8)               # (out, n_super, 128)
    qh = blocks[..., 128:128 + 64].astype(mx.uint8)        # (out, n_super, 64)
    scales_u8 = blocks[..., 192:208].astype(mx.uint8)      # (out, n_super, 16)
    # scales are signed int8 in the real struct -- reinterpret, not cast.
    scales = mx.where(scales_u8 >= 128,
                       scales_u8.astype(mx.float32) - 256.0,
                       scales_u8.astype(mx.float32))       # (out, n_super, 16)
    d = _fp16_bytes_to_f32(blocks[..., 208:210])           # (out, n_super)

    # Real dequantize_row_q6_K: 2 outer iterations (n=0,128), each handling
    # 128 output values from ql[64 bytes]/qh[32 bytes]/8 of the 16 scales.
    # Critically, within each outer iteration `is = l/16` (l=0..31) itself
    # flips mid-group: the first 16 of each 32-wide q1..q4 group use one
    # scale, the last 16 use the next one -- so each 32-wide group is
    # actually two 16-wide sub-blocks, matching QK_K/16=16 scales total.
    def _two_scale_halves(bits, sc_lo, sc_hi):
        # bits: (out, n_super, 32); first 16 cols use sc_lo, last 16 use sc_hi.
        lo = bits[..., :16] * sc_lo[..., None]
        hi = bits[..., 16:] * sc_hi[..., None]
        return mx.concatenate([lo, hi], axis=-1)  # (out, n_super, 32)

    outputs = [None, None]
    for outer in range(2):  # n = 0, 128
        ql_o = ql[..., outer * 64:(outer + 1) * 64].reshape(out_features, n_super, 2, 32)
        qh_o = qh[..., outer * 32:(outer + 1) * 32]  # (out, n_super, 32)
        is_base = outer * 8
        sc = [scales[..., is_base + k] for k in range(8)]  # each (out, n_super)
        ql_lo, ql_hi = ql_o[..., 0, :], ql_o[..., 1, :]  # (out, n_super, 32) each
        q1 = ((ql_lo & 0xF) | (((qh_o >> 0) & 3) << 4)).astype(mx.float32) - 32.0
        q2 = ((ql_hi & 0xF) | (((qh_o >> 2) & 3) << 4)).astype(mx.float32) - 32.0
        q3 = ((ql_lo >> 4) | (((qh_o >> 4) & 3) << 4)).astype(mx.float32) - 32.0
        q4 = ((ql_hi >> 4) | (((qh_o >> 6) & 3) << 4)).astype(mx.float32) - 32.0
        d_b = d[..., None]  # (out, n_super, 1)
        y1 = d_b * _two_scale_halves(q1, sc[0], sc[1])  # (out, n_super, 32)
        y2 = d_b * _two_scale_halves(q2, sc[2], sc[3])
        y3 = d_b * _two_scale_halves(q3, sc[4], sc[5])
        y4 = d_b * _two_scale_halves(q4, sc[6], sc[7])
        outputs[outer] = mx.concatenate([y1, y2, y3, y4], axis=-1)  # (out, n_super, 128)
    y = mx.concatenate(outputs, axis=-1)  # (out, n_super, 256)
    return y.reshape(out_features, in_features).astype(out_dtype)


# ---- DeepSeek V4 block-scaled formats (F204) -------------------------------
#
# The released DeepSeek-V4-Flash-0731 checkpoint uses TWO distinct schemes, and
# neither config.json nor the HuggingFace API describes the important one.
# config.json declares ``quantization_config.quant_method = "fp8"`` and the API
# summary says the experts are FP4; the real headers say the 35,328 routed
# expert weight tensors are ``I8`` with ``F8_E8M0`` scales. Trusting either
# source would have produced a path that runs and silently returns garbage on
# every routed expert -- the same class of error F93 and F128 caught on K2.5
# and K3. Verify tensor dtypes against the real checkpoint, always.
#
# Both schemes share the E8M0 scale: an 8-bit biased exponent whose value is
# 2**(e - 127), the same convention K3's expert scales use.


def decode_e8m0_scale(raw: mx.array) -> mx.array:
    """Decode E8M0 power-of-two scale bytes to float32.

    Built from the IEEE-754 bit pattern rather than ``2.0 ** (e - 127)``:
    exponent byte ``0`` denotes 2**-127, which is *subnormal* in float32, and
    the arithmetic form flushes it to zero. That would silently zero every
    weight block carrying the smallest scale -- caught by comparing all 256
    encodings against ml_dtypes, not a sample.

    For ``e`` in 1..254 the float32 exponent field is exactly ``e``, so the
    pattern is ``e << 23``. ``0`` is the subnormal ``0x00400000`` and ``255``
    is the format's NaN.
    """
    if raw.dtype != mx.uint8:
        raise ValueError(f"E8M0 scales must be uint8, got {raw.dtype}")
    wide = raw.astype(mx.uint32)
    bits = wide << 23
    bits = mx.where(wide == 0, mx.array(0x00400000, mx.uint32), bits)
    bits = mx.where(wide == 255, mx.array(0x7FC00000, mx.uint32), bits)
    return bits.view(mx.float32)


def _broadcast_block_scale(scale: mx.array, shape: tuple[int, ...]
                           ) -> mx.array:
    """Expand a per-block scale to full weight shape by exact repetition.

    Blocking is inferred from the ratio of weight to scale extent on each axis
    and must divide exactly; a non-integral ratio means the caller has paired
    the wrong scale with the weight, which is exactly the failure that would
    otherwise dequantize to plausible noise.
    """
    if scale.ndim != len(shape):
        raise ValueError(
            f"scale rank {scale.ndim} does not match weight rank {len(shape)}")
    for axis, (full, blocks) in enumerate(zip(shape, scale.shape)):
        if blocks <= 0 or full % blocks:
            raise ValueError(
                f"axis {axis}: weight extent {full} is not an exact multiple "
                f"of scale extent {blocks}")
        scale = mx.repeat(scale, full // blocks, axis=axis)
    return scale


def dequantize_deepseek_v4_fp8(packed: mx.array, scale: mx.array) -> mx.array:
    """Dequantize an E4M3 weight with E8M0 block scales (attention/shared).

    MLX decodes E4M3 natively from the uint8 payload safetensors stores, so no
    hand-written bit unpacking is involved.
    """
    if packed.dtype != mx.uint8:
        raise ValueError(
            f"DeepSeek V4 FP8 weights must load as uint8, got {packed.dtype}")
    values = mx.from_fp8(packed, mx.float32)
    decoded = decode_e8m0_scale(scale)
    if decoded.ndim == 2 and values.ndim == 2:
        # Expose the block axes and let MLX broadcast, instead of repeating the
        # scale up to full weight shape first. mx.repeat MATERIALIZES that
        # expansion -- 67MB of float32 per 4096x4096 trunk tensor, per layer,
        # per token, purely to be multiplied away. Bit-identical, 1.38x faster
        # (3.36 -> 4.63GB/s of raw input on a real trunk shape).
        rows, cols = values.shape
        bm, bn = rows // decoded.shape[0], cols // decoded.shape[1]
        if bm * decoded.shape[0] == rows and bn * decoded.shape[1] == cols:
            blocked = values.reshape(decoded.shape[0], bm, decoded.shape[1], bn)
            return (blocked * decoded[:, None, :, None]).reshape(
                rows, cols).astype(mx.bfloat16)
    return (values * _broadcast_block_scale(
        decoded, values.shape)).astype(mx.bfloat16)


def dequantize_finegrained_fp8(
        packed: mx.array, weight_scale_inv: mx.array, *,
        block_shape: tuple[int, int] | None = None) -> mx.array:
    """Dequantize HF fine-grained E4M3 weights with float32 block scales.

    Both released GLM-5.3 checkpoints store every converted matrix as
    ``weight`` plus
    ``weight_scale_inv``.  Despite the historical name, the sibling contains
    the *dequantization multiplier*: the official Transformers conversion is
    ``float8(weight) * weight_scale_inv`` over the released 128x128 grid.
    This is distinct from DeepSeek-V4's E8M0 exponent-byte ``.scale`` format.
    Keeping separate entry points makes mixing those two plausible-looking
    encodings a hard error instead of silent numerical corruption.
    """
    if packed.dtype != mx.uint8:
        raise ValueError(
            f"fine-grained FP8 weights must load as uint8, got {packed.dtype}")
    if weight_scale_inv.dtype != mx.float32:
        raise ValueError(
            "fine-grained FP8 weight_scale_inv must be float32, got "
            f"{weight_scale_inv.dtype}")
    values = mx.from_fp8(packed, mx.float32)
    if block_shape is not None:
        if (len(block_shape) != values.ndim
                or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
                       for v in block_shape)):
            raise ValueError(
                "fine-grained FP8 block_shape must contain one positive "
                "integer per weight axis")
        expected = tuple(
            (int(full) + int(block) - 1) // int(block)
            for full, block in zip(values.shape, block_shape)
        )
        if tuple(weight_scale_inv.shape) != expected:
            raise ValueError(
                "fine-grained FP8 scale grid does not match the declared "
                f"block shape: weight={tuple(values.shape)}, "
                f"scale={tuple(weight_scale_inv.shape)}, "
                f"block={tuple(block_shape)}, expected={expected}")
        if values.ndim == 2:
            rows, cols = values.shape
            block_rows, block_cols = block_shape
            padded_rows = int(weight_scale_inv.shape[0]) * block_rows
            padded_cols = int(weight_scale_inv.shape[1]) * block_cols
            padded = values
            if padded_rows != rows or padded_cols != cols:
                padded = mx.pad(
                    padded,
                    ((0, padded_rows - rows), (0, padded_cols - cols)),
                )
            blocked = padded.reshape(
                weight_scale_inv.shape[0], block_rows,
                weight_scale_inv.shape[1], block_cols)
            decoded = (blocked * weight_scale_inv[:, None, :, None]).reshape(
                padded_rows, padded_cols)
            return decoded[:rows, :cols].astype(mx.bfloat16)
        # The Hub format ceil-divides its scale grid.  A dimension such as
        # 576 therefore owns five 128-row scale blocks, with only 64 rows in
        # the final block. Repeat by the declared block extent and crop the
        # padded tail; inferring 576/5 would silently use the wrong 116-row
        # boundaries. Two-dimensional model weights take the padded blocked
        # path above, so this full-size scale expansion remains only a generic
        # fallback for a future higher-rank fine-grained tensor.
        expanded = weight_scale_inv
        for axis, block in enumerate(block_shape):
            expanded = mx.repeat(expanded, int(block), axis=axis)
        slices = tuple(slice(0, int(full)) for full in values.shape)
        return (values * expanded[slices]).astype(mx.bfloat16)
    if weight_scale_inv.ndim == 2 and values.ndim == 2:
        rows, cols = values.shape
        scale_rows, scale_cols = weight_scale_inv.shape
        if (scale_rows > 0 and scale_cols > 0
                and rows % scale_rows == 0 and cols % scale_cols == 0):
            block_rows = rows // scale_rows
            block_cols = cols // scale_cols
            blocked = values.reshape(
                scale_rows, block_rows, scale_cols, block_cols)
            return (blocked * weight_scale_inv[:, None, :, None]).reshape(
                rows, cols).astype(mx.bfloat16)
    return (values * _broadcast_block_scale(
        weight_scale_inv, values.shape)).astype(mx.bfloat16)


def dequantize_deepseek_v4_fp4(packed: mx.array, scale: mx.array
                               ) -> mx.array:
    """Dequantize a routed expert: E2M1 FP4 packed two per byte, E8M0 scales.

    CORRECTION (supersedes the INT8 reading in F204). The safetensors header
    says ``I8``, which describes the storage container, not the contents: each
    byte holds TWO 4-bit codes, so the logical width is twice the stored one.
    Three independent checks agree -- ``w1`` unpacks to (2048, 4096) whose
    in-features match hidden_size 4096 and whose out-features match
    moe_intermediate_size 2048 (the stored shape chains with neither); the
    stored byte count is exactly half the logical value count; and the scale
    granularity works out to 32 logical values per scale, matching the
    released ``fp4_block_size``. config.json's ``expert_dtype: "fp4"`` was
    right, and the earlier reading of the header over the config was wrong.

    The F204 INT8 test passed because it compared against a numpy
    reimplementation of the same assumption -- self-referential on exactly the
    packing question. Only running a real forward pass surfaced it.

    Codes are the OCP E2M1 set, low nibble first, identical to the MXFP4 path
    K3 already uses; the difference here is the int8 container and the
    per-row/32-column scale blocking.
    """
    if packed.dtype not in (mx.int8, mx.uint8):
        raise ValueError(
            f"DeepSeek V4 routed experts must load as int8/uint8, got "
            f"{packed.dtype}")
    raw = packed.view(mx.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    # Interleave so byte i yields logical columns 2i (low) and 2i+1 (high).
    codes = mx.stack([low, high], axis=-1).reshape(
        raw.shape[0], raw.shape[1] * 2)

    lut = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], mx.float32)
    magnitude = lut[codes & 0x07]
    values = mx.where((codes & 0x08) != 0, -magnitude, magnitude)

    return values * _broadcast_block_scale(
        decode_e8m0_scale(scale), values.shape)


def dequantize_deepseek_v4_int8(packed: mx.array, scale: mx.array) -> mx.array:
    """Dequantize a symmetric INT8 routed expert with E8M0 block scales.

    Symmetric: the checkpoint ships no zero-point tensor, so the value is
    ``int8 * 2**(e - 127)`` with no offset. Routed experts block only along the
    column axis (weight ``[2048, 2048]`` against scale ``[2048, 128]`` is one
    scale per row per 16 columns), unlike the 128x128 blocking the dense path
    uses -- so the blocking is derived per tensor rather than assumed.
    """
    if packed.dtype != mx.int8:
        raise ValueError(
            f"DeepSeek V4 routed experts must load as int8, got {packed.dtype}")
    values = packed.astype(mx.float32)
    return values * _broadcast_block_scale(
        decode_e8m0_scale(scale), values.shape)
