"""Quantize-on-load: weights are stored on disk at full precision (bf16/fp16) and
optionally quantized as they enter the WeightCache. Disk reads stay full-precision;
the *resident* footprint shrinks 4-8x, which lets far more (often all) layers stay
cached. This trades quantization error for residency — configurable per module so
attention can stay bf16 while the MLP goes 4-bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


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
    This representation is therefore restricted to an explicitly lossy profile
    and keeps the exact head resident for reranking.
    """

    exact: mx.array
    approx: QTensor
    candidates: int

    @property
    def nbytes(self) -> int:
        return self.exact.nbytes + self.approx.nbytes

    @property
    def shape(self) -> tuple[int, ...]:
        return self.exact.shape

    @property
    def dtype(self):
        return self.exact.dtype


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
        if name.endswith(".mlp.gate.weight") and not self.quantize_router:
            return False
        if ".mlp." in name:
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


def matmul(x: mx.array, w) -> mx.array:
    """x @ w.T for a plain, quantized, or candidate-reranked weight."""
    if isinstance(w, RerankedQHead):
        approx = matmul(x, w.approx)
        k = w.candidates
        indices = mx.argpartition(
            -approx, kth=k - 1, axis=-1)[..., :k]

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

        sparse = mx.full(
            approx.shape, float("-inf"), dtype=approx.dtype)
        return mx.put_along_axis(
            sparse, indices, exact_scores.astype(approx.dtype), axis=-1)
    if isinstance(w, QTensor):
        return mx.quantized_matmul(
            x, w.wq, scales=w.scales, biases=w.biases,
            transpose=True, group_size=w.group_size, bits=w.bits, mode=w.mode,
        )
    return x @ w.T
