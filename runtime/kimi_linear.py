"""Kimi Linear (KDA hybrid) block math -- Kimi K3 readiness prep.

See docs/future_lossless_techniques.md F92 for the full architecture audit
of the real moonshotai/Kimi-Linear-48B-A3B-Instruct checkpoint, the
implementation plan, and open gates.

VERIFIED against a real numerical oracle as of 2026-07-18
(tests/test_f92_kda_oracle.py): the KDA attention block, the MLA attention
block (NoPE variant), and the MoE gate+expert routing all match the real,
unmodified `modeling_kimi.py` to <1e-3 max abs diff on a tiny random-weight
instance (same methodology as tests/test_f33_mla_attention.py for GLM --
extract a real HF module's state_dict, feed identical weights through this
module, compare outputs). `fla-core`'s ops package unconditionally imports
`triton` at package-init time and there is no Triton wheel for Apple Silicon
macOS, so the oracle test installs pure-PyTorch stand-ins (formulas
transcribed from the real fla-org/flash-linear-attention source, not
reconstructed from memory) for exactly the pieces `fla` would have supplied,
then runs the real released model code around them. This does NOT use the
real 48B-parameter released weights (infeasible to instantiate as PyTorch
nn.Parameters on this machine's RAM) -- see tests/test_kimi_linear_smoke.py
for the separate real-weights shape/plumbing smoke test.

The oracle caught two real bugs this module's first version got wrong:
1. Kimi Linear's MLA is NoPE (`config.mla_use_nope=True`) -- the real
   `KimiMLAAttention.forward` never calls any rotary-embedding function at
   all; position information comes only from the KDA layers' inherent
   sequential recurrence. `runtime.glm._mla_attention` gained an
   `mla_use_nope` branch for this (GLM always applies real RoPE, unaffected).
2. Kimi's real `KimiMoEGate.forward` has `scores_for_choice = scores.view(...);
   scores_for_choice += bias` -- an in-place `+=` on a `.view()`, which
   aliases and mutates the original `scores` tensor. So the released model's
   actual executed routing WEIGHT (not just expert selection) is computed
   from the bias-corrected score, unlike GLM's noaux_tc design where bias
   affects selection only. Verified to 6 decimal places against the real
   gate before fixing `_route_experts` below to match -- this is very
   likely an unintentional aliasing bug in the released reference code, but
   the mission is byte-for-byte replication of AS-RELEASED behavior, not
   presumed design intent.

Architecture (from the real downloaded modeling_kimi.py / config.json):
- 27 layers. cfg.full_attn_layers (7 of 27, 0-indexed) use MLA; every other
  layer (20 of 27, cfg.kda_layers) uses KDA. Layer 0 is dense MLP (all other
  layers >=1 are MoE, first_k_dense_replace=1, moe_layer_freq=1).
- MLA layers are DeepSeek/GLM-5.2-shaped (kv_a_proj_with_mqa -> RMSNorm ->
  kv_b_proj, NoPE) but with q_lora_rank=null (no Q compression, a single
  q_proj) -- runtime.glm._mla_attention was generalized to handle both.
- MoE gate is the same noaux_tc sigmoid+bias-correction flat top-k as GLM's
  (n_group=topk_group=1 in this checkpoint) -- runtime.glm._route_experts /
  _group_routes are reused directly. Per-routed-expert MLPs use w1/w2/w3
  naming (w1=gate, w2=down, w3=up) instead of GLM's gate_proj/up_proj/
  down_proj, so a small local swiglu variant is used for them; the dense
  layer-0 MLP and each MoE layer's shared_experts both use the ordinary
  gate_proj/up_proj/down_proj naming and reuse layer_runner._swiglu.
- KDA (KimiDeltaAttention) recurrence, per this project's 2026-07-18 read of
  fla-org/flash-linear-attention's ops/kda/{gate,naive,fused_recurrent}.py
  (no local copy of that package to import from -- Triton/CUDA-only anyway):
    q, k are L2-normalized per (batch, head, timestep) over head_dim (eps
    1e-6), q additionally scaled by head_dim ** -0.5, both AFTER a causal
    depthwise conv1d (kernel_size=4, SiLU) applied to q/k/v.
    gate g = -exp(A_log) * softplus(f_b_proj(f_a_proj(h)) + dt_bias), shape
    (B, L, H, head_dim) -- a per-(head, key-channel) log-decay.
    beta = sigmoid(b_proj(h)), a per-head scalar (the delta-rule write
    strength).
    Per-timestep state update (S: (B, H, head_dim, head_dim), K-axis is the
    key/decay axis, V-axis is the value axis):
        S_t = S_{t-1} * exp(g_t)[..., None]
        pred_v = sum_K(k_t[..., None] * S_t)
        S_t = S_t + (beta_t * k_t)[..., None] * (v_t - pred_v)[..., None, :]
        o_t = sum_K(q_t[..., None] * S_t)
  This sequential-scan implementation is correctness-first, not the chunked-
  parallel algorithm the real kernel uses for speed (F92's explicit
  correctness-before-speed stop rule). It is O(L) Python-level steps; expect
  it to be impractically slow for anything beyond a short smoke-test prefix.
- Output: a second low-rank gate (g_a_proj -> g_b_proj) feeds a sigmoid-
  gated RMSNorm (normalize o, scale by o_norm.weight, THEN multiply by
  sigmoid(gate) -- gate applied AFTER normalization, per fla's
  modules/fused_norm_gate.py) before o_proj.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os
import tempfile

import mlx.core as mx
import numpy as np

from . import quant
from .config import ModelConfig, effective_expert_top_k
from .expert_batching import consume_expert_batches
from .glm import _group_routes, _mla_attention
from .kda_state import KDAStateCache
from .layer_runner import _linear, _swiglu
from .uncached_io import set_darwin_nocache


class _DiskAttnResSnapshot:
    """Exact BF16 row-addressable AttnRes snapshot stored on local disk."""

    ndim = 2

    def __init__(self, owner, index: int):
        self._owner = owner
        self.index = int(index)

    @property
    def path(self) -> Path:
        return self._owner._group_path(self.index // self._owner.group_size)

    @property
    def shape(self) -> tuple[int, int]:
        return self._owner.shape

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("disk AttnRes snapshots require contiguous slices")
        start, stop, step = key.indices(self.shape[0])
        if step != 1:
            raise TypeError("disk AttnRes snapshots require unit stride")
        return self._owner.read_stacked(
            start, stop, count=self.index + 1
        )[:, self.index, :]


class DiskBackedAttnResSnapshots(list):
    """List-compatible bounded-memory store for K3 residual snapshots.

    AttnRes only appends one immutable BF16 ``[positions, hidden]`` snapshot
    every 12 layers, then consumes contiguous row tiles. Persisting each
    snapshot in exact released dtype avoids a multi-gigabyte Metal residency
    floor at 46K+ tokens while preserving every value byte. The directory is
    temporary and removed after the sweep.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        write_tile_rows: int = 256,
        group_size: int = 4,
    ):
        super().__init__()
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="voom-k3-attnres-", dir=root)
        self.directory = Path(self._temporary.name)
        self.group_size = max(1, int(group_size))
        # Keep the historical attribute for diagnostics/tests.  It is the
        # first bounded packed group, not an unbounded all-snapshot file.
        self._packed_path = self._group_path(0)
        self._readers: dict[int, object] = {}
        self.shape: tuple[int, int] = (0, 0)
        self.write_tile_rows = max(1, int(write_tile_rows))
        self.bytes_written = 0
        self.bytes_read = 0
        self.write_calls = 0
        self.read_calls = 0
        self.uncached_descriptors = 0

    def _group_path(self, group: int) -> Path:
        return self.directory / f"snapshots-{int(group):03d}.bf16"

    def _close_reader(self, group: int) -> None:
        reader = self._readers.pop(int(group), None)
        if reader is not None:
            reader.close()

    def append(self, value):
        if value.ndim != 2 or value.dtype != mx.bfloat16:
            raise ValueError(
                "disk AttnRes snapshots require rank-2 BF16 values")
        rows, hidden = map(int, value.shape)
        old_blocks = len(self)
        if old_blocks and self.shape != (rows, hidden):
            raise ValueError(
                f"AttnRes snapshot shape {(rows, hidden)} != {self.shape}")
        group = old_blocks // self.group_size
        old_group_blocks = old_blocks % self.group_size
        packed_path = self._group_path(group)
        temporary_path = self.directory / f"snapshots-{group:03d}.next.bf16"
        self._close_reader(group)
        old_source = (
            packed_path.open("rb", buffering=0)
            if old_group_blocks else None
        )
        with temporary_path.open("wb", buffering=0) as output:
            self.uncached_descriptors += int(
                set_darwin_nocache(output.fileno()))
            if old_source is not None:
                self.uncached_descriptors += int(
                    set_darwin_nocache(old_source.fileno()))
            for start in range(0, rows, self.write_tile_rows):
                end = min(start + self.write_tile_rows, rows)
                tile = value[start:end]
                mx.eval(tile)
                new_host = np.asarray(tile.view(mx.uint16))
                if new_host.dtype != np.uint16:
                    raise TypeError(
                        f"AttnRes host dtype {new_host.dtype} is not uint16")
                if old_source is not None:
                    old_elements = (
                        (end - start) * old_group_blocks * hidden
                    )
                    old_payload = old_source.read(old_elements * 2)
                    if len(old_payload) != old_elements * 2:
                        raise IOError(
                            "short packed AttnRes read while appending: "
                            f"{len(old_payload)} != {old_elements * 2}")
                    old_host = np.frombuffer(
                        old_payload, dtype=np.uint16
                    ).reshape(end - start, old_group_blocks, hidden)
                    combined = np.concatenate(
                        [old_host, new_host[:, None, :]], axis=1)
                    self.bytes_read += len(old_payload)
                    self.read_calls += 1
                else:
                    combined = new_host[:, None, :]
                payload = combined.tobytes(order="C")
                output.write(payload)
                self.bytes_written += len(payload)
                self.write_calls += 1
            output.flush()
            os.fsync(output.fileno())
        if old_source is not None:
            old_source.close()
        expected = rows * (old_group_blocks + 1) * hidden * 2
        if temporary_path.stat().st_size != expected:
            raise IOError(
                "packed AttnRes size "
                f"{temporary_path.stat().st_size} != {expected}")
        os.replace(temporary_path, packed_path)
        self.shape = (rows, hidden)
        self.clear()
        for index in range(old_blocks + 1):
            super().append(_DiskAttnResSnapshot(self, index))
        self._readers[group] = packed_path.open("rb", buffering=0)
        self.uncached_descriptors += int(set_darwin_nocache(
            self._readers[group].fileno()))

    def read_stacked(
        self, start: int, stop: int, *, count: int | None = None,
    ) -> mx.array:
        """Read one contiguous row stripe across every requested snapshot."""
        blocks = len(self)
        count = blocks if count is None else int(count)
        if not 0 <= count <= blocks:
            raise ValueError(f"AttnRes count {count} is outside [0, {blocks}]")
        start, stop, step = slice(start, stop).indices(self.shape[0])
        if step != 1:
            raise TypeError("packed AttnRes requires unit row stride")
        rows = stop - start
        if count == 0:
            return mx.zeros(
                (rows, 0, self.shape[1]), dtype=mx.bfloat16)
        pieces = []
        for group, first in enumerate(range(0, count, self.group_size)):
            stored = min(self.group_size, blocks - first)
            requested = min(stored, count - first)
            elements = rows * stored * self.shape[1]
            offset = start * stored * self.shape[1] * 2
            path = self._group_path(group)
            reader = self._readers.get(group)
            if reader is None:
                reader = path.open("rb", buffering=0)
                self.uncached_descriptors += int(
                    set_darwin_nocache(reader.fileno()))
                self._readers[group] = reader
            reader.seek(offset)
            payload = reader.read(elements * 2)
            if len(payload) != elements * 2:
                raise IOError(
                    f"short packed AttnRes read from {path}: "
                    f"{len(payload)} != {elements * 2}")
            host = np.frombuffer(payload, dtype=np.uint16).reshape(
                rows, stored, self.shape[1]
            )
            if requested != stored:
                host = np.ascontiguousarray(host[:, :requested, :])
            pieces.append(host)
            self.bytes_read += len(payload)
            self.read_calls += 1
        host = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)
        return mx.array(host).view(mx.bfloat16)

    def stats(self) -> dict[str, int]:
        return {
            "snapshots": len(self),
            "bytes_written": self.bytes_written,
            "bytes_read": self.bytes_read,
            "write_calls": self.write_calls,
            "read_calls": self.read_calls,
            "uncached_descriptors": self.uncached_descriptors,
        }

    def close(self) -> None:
        self.clear()
        for group in list(self._readers):
            self._close_reader(group)
        self._temporary.cleanup()


def _route_experts(
        h: mx.array, w: dict, moe_prefix: str, cfg: ModelConfig,
        layer: int | None = None) -> tuple[mx.array, mx.array]:
    """Kimi's MoE router. NOT the same weight math as runtime.glm._route_experts.

    Gate weight path differs from GLM's hardcoded f"{prefix}.mlp.gate.*"
    (Kimi's MoE module lives under f"{prefix}.block_sparse_moe.gate.*"), so
    this is a local duplicate rather than a reparametrized import -- avoids
    touching glm._route_experts's existing call sites
    (tests/test_f33_router_oracle.py calls it directly).

    `layer` (optional, defaults to None -- every existing call site that
    doesn't pass it keeps its exact prior behavior) selects two explicit,
    lossy K3 policies when configured: `expert_prune_masks` can exclude
    calibrated REAP-style expert IDs, and `expert_top_k_by_layer` can lower
    the released routed budget uniformly or by layer. Both are numeric
    runtime schedules independent of prompt/tool/subject content. With both
    schedules unset (the default for every checkpoint), this is a
    byte-for-byte no-op.

    F92 oracle finding (2026-07-18, real modeling_kimi.py, verified to 6
    decimal places against the actual released KimiMoEGate): unlike GLM's
    noaux_tc design where the bias affects ONLY which experts are selected,
    Kimi's real released gate computes
        scores_for_choice = scores.view(...); scores_for_choice += bias
    -- an in-place `+=` on a `.view()`, which ALIASES and mutates the
    original `scores` tensor too. So by the time the real code does
    `topk_weight = scores.gather(1, topk_idx)`, `scores` has ALREADY been
    bias-corrected -- the routing WEIGHT (not just the selection) is
    computed from the biased score. This is very likely an unintentional
    aliasing bug in the released reference code, not deliberate design (it
    contradicts the whole point of noaux_tc bias-correction), but this
    project's mission is byte-for-byte replication of the AS-RELEASED
    checkpoint's actual behavior, not the presumed design intent -- so `pw`
    below is deliberately gathered from `biased`, not `scores`. Do not
    "fix" this to look like GLM's version.
    """
    gate_weight = w[f"{moe_prefix}.gate.weight"]
    from .bf16_nf12_linear import NF12Tensor

    if isinstance(gate_weight, (quant.QTensor, NF12Tensor)):
        router_logits = quant.matmul(h.astype(mx.float32), gate_weight)
    else:
        router_logits = h.astype(mx.float32) @ gate_weight.astype(mx.float32).T
    scores = mx.sigmoid(router_logits)
    biased = scores + w[f"{moe_prefix}.gate.e_score_correction_bias"]
    if (cfg.model_type == "kimi_k3" and layer is not None
            and cfg.expert_prune_masks and layer in cfg.expert_prune_masks):
        pruned = cfg.expert_prune_masks[layer]
        penalty = [0.0] * biased.shape[-1]
        for e in pruned:
            penalty[e] = -1e9
        biased = biased + mx.array(penalty, dtype=mx.float32)
    k = (
        effective_expert_top_k(cfg, layer)
        if layer is not None
        else cfg.num_experts_per_tok
    )
    idx = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k]
    if cfg.model_type == "kimi_k3":
        # F128: K3's real bundled modeling_kimi_linear.py FIXED this aliasing
        # bug -- its KimiMoEGate.forward computes
        #     scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        # using regular `+` (a fresh tensor), never `+=` on a `.view()`, so
        # `scores` itself is never mutated; `topk_weight = scores.gather(...)`
        # genuinely reads the UNBIASED scores -- bias affects selection only,
        # matching GLM's noaux_tc design intent. Confirmed by directly
        # reading the real, bundled modeling_kimi_linear.py shipped with the
        # actual downloaded K3 checkpoint (not re-derived/assumed from the
        # original Kimi Linear 48B's own real aliasing bug below).
        pw = mx.take_along_axis(scores, idx, axis=-1)
    else:
        pw = mx.take_along_axis(biased, idx, axis=-1)  # F92: biased, not scores -- see docstring
    if cfg.norm_topk_prob:
        pw = pw / (pw.sum(axis=-1, keepdims=True) + 1e-20)
    pw = pw * cfg.routed_scaling_factor
    return idx, pw


def _causal_depthwise_conv1d(
    x: mx.array, weight: mx.array, history: mx.array | None, kernel_size: int,
) -> tuple[mx.array, mx.array]:
    """Per-channel causal conv (PyTorch Conv1d cross-correlation, no flip), SiLU-activated.

    x: (B, L, C). weight: (C, 1, K) HF Conv1d layout. history: (B, K-1, C)
    carried from a previous call, or None (zero-padded) for the first call.
    Returns (silu(conv(x)), new_history).
    """
    B, L, C = x.shape
    K = kernel_size
    if history is None:
        history = mx.zeros((B, K - 1, C), dtype=x.dtype)
    padded = mx.concatenate([history, x], axis=1)  # (B, L+K-1, C)
    taps = weight.reshape(C, K)  # (C, K), tap k=K-1 is the current timestep
    out = mx.zeros((B, L, C), dtype=mx.float32)
    for k in range(K):
        out = out + padded[:, k:k + L, :].astype(mx.float32) * taps[:, k].astype(mx.float32)
    new_history = padded[:, L:, :] if K > 1 else mx.zeros((B, 0, C), dtype=x.dtype)
    activated = (mx.sigmoid(out) * out).astype(x.dtype)
    return activated, new_history


def _gated_rms_norm(x: mx.array, gate: mx.array, weight: mx.array, eps: float) -> mx.array:
    """out = rmsnorm(x) * weight * sigmoid(gate) -- gate applied AFTER norm+scale.

    F105-style native-primitive reuse (2026-07-25): this was a hand-rolled
    composite (mean/sqrt/multiply) that never called mx.fast.rms_norm,
    unlike this same file's input_layernorm/post_attention_layernorm calls
    a few lines below, which already did. No formula transform needed here
    (plain weight scaling, no zero-centered offset) -- verified
    byte-identical (0.0 max abs diff) against the original composite.
    Same expectation as F105's qwen3.5 case: this op is tiny relative to
    the matmul/disk costs that dominate a decode step, so it is kept as a
    correctness-preserving simplification, not claimed as a speed win
    without a real measurement to back that claim."""
    source_dtype = x.dtype
    x32 = x.astype(mx.float32)
    w32 = weight.astype(mx.float32)
    normed = mx.fast.rms_norm(x32, w32, eps)
    return (normed * mx.sigmoid(gate.astype(mx.float32))).astype(source_dtype)


def _kimi_expert_swiglu(h: mx.array, w: dict, prefix: str) -> mx.array:
    """Per-routed-expert MLP: w1=gate, w2=down, w3=up (Mixtral-style naming)."""
    gate = _linear(h, w, f"{prefix}.w1")
    up = _linear(h, w, f"{prefix}.w3")
    activated = mx.sigmoid(gate) * gate * up
    return _linear(activated, w, f"{prefix}.w2")


def _situ_and_mul(gate: mx.array, up: mx.array, beta: float, linear_beta: float) -> mx.array:
    """Kimi K3's real "situ" activation (hidden_act="situ"), verbatim from the
    real modeling_kimi_linear.py's SituAndMul.forward: both halves are
    upcast to float32 for the tanh/sigmoid, result cast back to the input
    dtype. `linear_beta` of 0.0 means "unset" (`up` passes through
    untransformed), matching the real code's `if self.linear_beta is not
    None`. NOT used by Kimi Linear 48B or Kimi K2.5 (both plain swiglu,
    cfg.hidden_act defaults to "silu" for them) -- Kimi K3 only."""
    gate32 = gate.astype(mx.float32)
    up32 = up.astype(mx.float32)
    situ_a = beta * mx.tanh(gate32 / beta) * mx.sigmoid(gate32)
    if linear_beta:
        up32 = linear_beta * mx.tanh(up32 / linear_beta)
    return (situ_a * up32).astype(gate.dtype)


def _kimi_expert_mlp(h: mx.array, w: dict, prefix: str, cfg: ModelConfig) -> mx.array:
    """Per-routed-expert MLP, dispatching activation by cfg.hidden_act.
    Kimi Linear/K2.5 (hidden_act="silu") take the exact same swiglu path as
    _kimi_expert_swiglu above (kept separate so that function's existing
    real-oracle test import/call sites are untouched); Kimi K3
    (hidden_act="situ") uses _situ_and_mul instead."""
    if cfg.hidden_act != "situ":
        return _kimi_expert_swiglu(h, w, prefix)
    gate = _linear(h, w, f"{prefix}.w1")
    up = _linear(h, w, f"{prefix}.w3")
    activated = _situ_and_mul(
        gate, up, cfg.activation_situ_beta, cfg.activation_situ_linear_beta)
    return _linear(activated, w, f"{prefix}.w2")


def _kimi_dense_mlp(
    h: mx.array,
    w: dict,
    prefix: str,
    cfg: ModelConfig,
    *,
    synchronize_subprojections: bool = False,
) -> mx.array:
    """gate_proj/up_proj/down_proj MLP (dense layer-0, and shared_experts),
    dispatching activation by cfg.hidden_act. Kimi Linear/K2.5 take the
    exact same path as the existing layer_runner._swiglu call sites (kept
    unchanged so their own real-oracle tests are untouched); Kimi K3 uses
    _situ_and_mul instead."""
    if cfg.hidden_act != "situ":
        return _swiglu(h, w, prefix)
    gate = _linear(h, w, f"{prefix}.gate_proj")
    if synchronize_subprojections:
        # K3's native MXFP4 gate/up/down projections are independent weight
        # decodes around tiny one-token activations.  Letting all three remain
        # in one lazy graph made their full-weight staging overlap at a measured
        # 6.5 GB.  Materializing each released projection in sequence preserves
        # every dot product while bounding staging to one projection at a time.
        mx.eval(gate)
    up = _linear(h, w, f"{prefix}.up_proj")
    if synchronize_subprojections:
        mx.eval(up)
    activated = _situ_and_mul(
        gate, up, cfg.activation_situ_beta, cfg.activation_situ_linear_beta)
    if synchronize_subprojections:
        mx.eval(activated)
    result = _linear(activated, w, f"{prefix}.down_proj")
    if synchronize_subprojections:
        mx.eval(result)
    return result


def _kimi_dense_mlp_tiled(
    h: mx.array,
    w: dict,
    prefix: str,
    cfg: ModelConfig,
    tile_size: int,
) -> mx.array:
    """Evaluate a row-independent dense Kimi MLP in bounded position tiles.

    K3's first layer has intermediate width 33,792. At 46K positions, keeping
    gate and up projections for the complete prompt would require more than
    6GB before the down projection. Position tiling preserves every row's
    released MLP equation and reduction dimension while retaining one loaded
    weight page for the whole layer-stationary sweep.
    """
    if tile_size <= 0 or h.shape[1] <= tile_size:
        return _kimi_dense_mlp(h, w, prefix, cfg)
    output = mx.zeros(
        (h.shape[0], h.shape[1], cfg.hidden_size), dtype=h.dtype
    )
    mx.eval(output)
    for start in range(0, h.shape[1], tile_size):
        end = min(start + tile_size, h.shape[1])
        value = _kimi_dense_mlp(
            h[:, start:end, :], w, prefix, cfg
        )
        mx.eval(value)
        output[:, start:end, :] = value
        mx.eval(output)
    return output


_NATIVE_KDA_PREFILL_SCAN_SOURCE = r"""
    constexpr uint MAX_D = 256;

    const uint dv = thread_position_in_grid.x;
    const uint h = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const uint tid = thread_index_in_threadgroup;
    const uint L = q_shape[1];
    const uint H = q_shape[2];
    const uint D = q_shape[3];

    // K3 has Dk == Dv == 128. One threadgroup owns one head, so load the
    // query/key/decay vectors once per position rather than once per value
    // column. The recurrent matrix itself stays in coherent global memory.
    threadgroup float shared_q[MAX_D];
    threadgroup float shared_k[MAX_D];
    threadgroup float shared_decay[MAX_D];

    const size_t state_base = ((size_t)b * H + h) * D * D + dv;
    for (uint t = 0; t < L; ++t) {
        const size_t vector_base = ((size_t)b * L + t) * H * D + h * D;
        if (tid < D) {
            shared_q[tid] = static_cast<float>(q[vector_base + tid]);
            shared_k[tid] = static_cast<float>(k[vector_base + tid]);
            shared_decay[tid] = exp(
                static_cast<float>(gate[vector_base + tid]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float predicted = 0.0f;
        for (uint dk = 0; dk < D; ++dk) {
            const size_t index = state_base + (size_t)dk * D;
            const float previous = t == 0
                ? static_cast<float>(state[index])
                : static_cast<float>(out_state[index]);
            const float decayed = previous * shared_decay[dk];
            out_state[index] = static_cast<T>(decayed);
            predicted += shared_k[dk] * decayed;
        }

        const size_t value_index = vector_base + dv;
        const float residual = static_cast<float>(v[value_index]) - predicted;
        const float scaled_residual = static_cast<float>(
            beta[((size_t)b * L + t) * H + h]) * residual;
        float output = 0.0f;
        for (uint dk = 0; dk < D; ++dk) {
            const size_t index = state_base + (size_t)dk * D;
            const float updated = static_cast<float>(out_state[index])
                + shared_k[dk] * scaled_residual;
            out_state[index] = static_cast<T>(updated);
            output += shared_q[dk] * updated;
        }
        out[value_index] = static_cast<T>(output);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


@lru_cache(maxsize=1)
def _native_kda_prefill_scan_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="voom_kimi_kda_prefill_scan",
        input_names=["q", "k", "v", "gate", "beta", "state"],
        output_names=["out", "out_state"],
        source=_NATIVE_KDA_PREFILL_SCAN_SOURCE,
    )


def _native_fused_kda_prefill_scan(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    gate: mx.array,
    beta: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    """Fuse an entire multi-position KDA recurrence into one Metal dispatch.

    This is the released recurrence, not a chunk/WY approximation: positions
    still advance serially and every state update precedes the corresponding
    output.  It changes the reduction schedule from MLX's separate ``sum``
    operations to a serial FP32 accumulation inside one kernel, so ordinary
    roundoff differences remain possible and serving must opt in until greedy
    released-model gates admit it.
    """
    if q.shape != k.shape or q.shape != v.shape or q.shape != gate.shape:
        raise ValueError(
            "native KDA prefill requires matching q/k/v/gate shapes")
    if len(q.shape) != 4:
        raise ValueError("native KDA prefill expects [B,L,H,D] tensors")
    batch, length, heads, dim = map(int, q.shape)
    if length <= 1:
        raise ValueError("native KDA prefill requires more than one position")
    if not 1 <= dim <= 256:
        raise ValueError("native KDA prefill head dimension must be <=256")
    if beta.shape != (batch, length, heads):
        raise ValueError(
            f"native KDA beta shape {beta.shape} != "
            f"{(batch, length, heads)}")
    if state.shape != (batch, heads, dim, dim):
        raise ValueError(
            f"native KDA state shape {state.shape} != "
            f"{(batch, heads, dim, dim)}")
    if any(value.dtype != mx.float32
           for value in (q, k, v, gate, beta, state)):
        raise ValueError("native KDA prefill currently requires float32")
    kernel = _native_kda_prefill_scan_kernel()
    if kernel is None:
        raise RuntimeError("native KDA prefill requires Metal")
    output, final_state = kernel(
        inputs=[q, k, v, gate, beta, state],
        template=[("T", state.dtype)],
        grid=(dim, heads, batch),
        threadgroup=(dim, 1, 1),
        output_shapes=[q.shape, state.shape],
        output_dtypes=[state.dtype, state.dtype],
    )
    return output, final_state


@mx.compile
def _compiled_kda_scan_segment(q, k, v, gate, beta, state):
    """Trace the ordinary MLX KDA recurrence without changing its operators."""
    outputs = []
    for position in range(q.shape[1]):
        q_t = q[:, position]
        k_t = k[:, position]
        v_t = v[:, position]
        state = state * mx.exp(gate[:, position])[..., None]
        predicted = mx.sum(k_t[..., None] * state, axis=-2)
        residual = v_t - predicted
        state = state + (
            beta[:, position, :, None] * k_t
        )[..., None] * residual[..., None, :]
        outputs.append(mx.sum(q_t[..., None] * state, axis=-2))
    return mx.stack(outputs, axis=1), state


def _compiled_kda_prefill_scan(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    gate: mx.array,
    beta: mx.array,
    state: mx.array,
    *,
    segment: int = 32,
) -> tuple[mx.array, mx.array]:
    """Compile the reference recurrence in bounded, byte-identical segments.

    Segment 32 deliberately retains the reference path's state-evaluation
    cadence.  The compiled graph uses the same MLX sums and elementwise ops;
    unlike the custom Metal scan above, it does not reassociate FP32 dots.
    """
    length = int(q.shape[1])
    if length <= 1 or segment <= 0:
        raise ValueError(
            "compiled KDA prefill requires multiple positions and a "
            "positive segment")
    outputs = []
    for start in range(0, length, segment):
        end = min(start + segment, length)
        output, state = _compiled_kda_scan_segment(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            gate[:, start:end],
            beta[:, start:end],
            state,
        )
        mx.eval(state)
        outputs.append(output)
    return mx.concatenate(outputs, axis=1), state


def _kda_attention(
    h: mx.array, w: dict, prefix: str, cfg: ModelConfig, kda_cache: KDAStateCache | None, layer: int,
    native_fused_decode: bool = False,
    native_fused_prefill: bool = False,
    compiled_prefill: bool = False,
    compiled_prefill_segment: int = 32,
    released_output_dtype: bool = False,
    profile=None,
) -> mx.array:
    B, L, _ = h.shape
    H = cfg.kda_num_heads
    D = cfg.kda_head_dim
    K = cfg.kda_conv_kernel_size

    projection_t0 = profile.start_substep() if profile is not None else None
    q = _linear(h, w, f"{prefix}.self_attn.q_proj")
    k = _linear(h, w, f"{prefix}.self_attn.k_proj")
    v = _linear(h, w, f"{prefix}.self_attn.v_proj")

    q_hist, k_hist, v_hist = (
        kda_cache.conv_history(layer) if kda_cache is not None and kda_cache.conv_history(layer) is not None
        else (None, None, None)
    )
    # 2026-07-25: KDA's per-channel causal conv (K-tap weighted sum + SiLU)
    # is mathematically IDENTICAL to qwen3.5's own conv1d+SiLU -- this file
    # already provides the plain/shared implementation qwen35.py imports
    # (_causal_depthwise_conv1d, above), and qwen35.py's F103 native fused
    # Metal kernel is a verified-byte-identical drop-in for it (same K-tap
    # weighted sum + sigmoid(acc)*acc SiLU, same (B,L,C)/(C,1,K) shapes).
    # Reused directly here, the same "found a mathematically identical
    # existing loop, reused the kernel unmodified" pattern that already
    # worked for Jet-Nemotron's DeltaNet-step kernel reuse.
    conv_fn = _causal_depthwise_conv1d
    if L == 1 and native_fused_decode:
        from .qwen35 import _native_fused_causal_conv1d
        conv_fn = _native_fused_causal_conv1d
    q, q_hist_new = conv_fn(q, w[f"{prefix}.self_attn.q_conv1d.weight"], q_hist, K)
    k, k_hist_new = conv_fn(k, w[f"{prefix}.self_attn.k_conv1d.weight"], k_hist, K)
    v, v_hist_new = conv_fn(v, w[f"{prefix}.self_attn.v_conv1d.weight"], v_hist, K)
    projection_dtype = q.dtype
    if profile is not None:
        profile.finish_substep(
            "kda_qkv_conv", layer, projection_t0, q, k, v,
            positions=L)

    gate_t0 = profile.start_substep() if profile is not None else None
    dt_bias = w[f"{prefix}.self_attn.dt_bias"].reshape(H, D).astype(mx.float32)
    g_raw = _linear(_linear(h, w, f"{prefix}.self_attn.f_a_proj"), w, f"{prefix}.self_attn.f_b_proj")
    g_raw = g_raw.reshape(B, L, H, D).astype(mx.float32) + dt_bias
    # F128: Kimi K3's real checkpoint saves A_log with head_dim elements
    # (128), not num_heads (96) -- confirmed directly against a real
    # downloaded layer's shard (b_proj/dt_bias both correctly reflect
    # num_heads=96 elsewhere in the SAME layer, so this is A_log-specific,
    # not a wrong num_heads reading). Fetched fla-org/flash-linear-
    # attention's real kda/gate.py Triton kernel source (2026-07-27) shows
    # A_log is indexed strictly `A_log + i_h` for i_h in [0, H) -- H passed
    # explicitly by the caller, never inferred from A_log's own tensor
    # size -- so the real kernel silently reads only the first H=96
    # elements regardless of the buffer's true (over-allocated) length.
    # Not runtime-verified against the real Triton kernel itself (no
    # CUDA/Triton on this machine), but this is the only interpretation
    # consistent with every real source available: the kernel's own
    # indexing, and the original Kimi Linear 48B's A_log (no over-
    # allocation, exactly num_heads elements, unaffected by this slice).
    A = mx.exp(w[f"{prefix}.self_attn.A_log"][:H].astype(mx.float32)).reshape(1, 1, H, 1)
    if cfg.kda_gate_lower_bound:
        # F128: Kimi K3's real linear_attn_config sets gate_lower_bound=-5.0
        # (safe_gate=True in the real KimiDeltaAttention.forward) -- ported
        # verbatim from the real kda_gate_fwd_kernel's USE_LOWER_BOUND
        # branch: `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, using
        # the RAW g_raw directly (no softplus at all in this branch, unlike
        # the no-lower-bound formula below).
        gate = cfg.kda_gate_lower_bound * mx.sigmoid(A * g_raw)
    else:
        softplus_g = mx.logaddexp(g_raw, mx.zeros_like(g_raw))  # log(1 + exp(x)), numerically stable
        gate = -A * softplus_g  # (B, L, H, D) log-decay, <= 0

    beta = mx.sigmoid(_linear(h, w, f"{prefix}.self_attn.b_proj").astype(mx.float32))  # (B, L, H)

    q = q.reshape(B, L, H, D).astype(mx.float32)
    k = k.reshape(B, L, H, D).astype(mx.float32)
    v = v.reshape(B, L, H, D).astype(mx.float32)

    def _l2norm(x):
        return x / mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-6)

    q = _l2norm(q) * (D ** -0.5)
    k = _l2norm(k)
    if profile is not None:
        profile.finish_substep(
            "kda_gate_norm", layer, gate_t0, q, k, v, gate, beta,
            positions=L)

    state = kda_cache.state(layer) if kda_cache is not None else None
    if state is None:
        state = mx.zeros((B, H, D, D), dtype=mx.float32)

    scan_t0 = profile.start_substep() if profile is not None else None
    if native_fused_prefill and compiled_prefill:
        raise ValueError(
            "native and compiled KDA prefill paths are mutually exclusive")
    if L > 1 and native_fused_prefill:
        o, state = _native_fused_kda_prefill_scan(
            q, k, v, gate, beta, state)
    elif L > 1 and compiled_prefill:
        o, state = _compiled_kda_prefill_scan(
            q, k, v, gate, beta, state,
            segment=compiled_prefill_segment)
    else:
        outputs = []
        for t in range(L):
            q_t, k_t, v_t, g_t, beta_t = q[:, t], k[:, t], v[:, t], gate[:, t], beta[:, t]
            state = state * mx.exp(g_t)[..., None]                       # (B,H,K,V) decay along K axis
            pred_v = mx.sum(k_t[..., None] * state, axis=-2)             # (B,H,V)
            residual = v_t - pred_v
            state = state + (beta_t[..., None] * k_t)[..., None] * residual[..., None, :]
            o_t = mx.sum(q_t[..., None] * state, axis=-2)                # (B,H,V)
            outputs.append(o_t)
            if (t + 1) % 32 == 0:
                # F92: bound the lazy graph -- a naive Python-level scan otherwise
                # accumulates one node per op per timestep with no eval boundary.
                mx.eval(state)
        o = mx.stack(outputs, axis=1)  # (B, L, H, D) float32
    if profile is not None:
        profile.finish_substep(
            "kda_scan", layer, scan_t0, o, state, positions=L)

    if kda_cache is not None:
        if kda_cache.factor_capture_active:
            if L != 1:
                raise ValueError(
                    "compact KDA factor capture requires serial positions")
            kda_cache.capture_factor_step(
                layer,
                gate=gate[:, 0],
                key=k[:, 0],
                value=v[:, 0],
                beta=beta[:, 0],
                conv_history=(q_hist_new, k_hist_new, v_hist_new),
            )
        mx.eval(state)
        kda_cache.set_state(layer, state)
        kda_cache.set_conv_history(layer, (q_hist_new, k_hist_new, v_hist_new))

    # F128: Kimi K3's real KimiDeltaAttention.forward picks a single
    # full-rank g_proj instead of the low-rank g_a_proj/g_b_proj split when
    # config.linear_attn_config.use_full_rank_gate is true (confirmed
    # present -- true -- on the real checkpoint; absent/false for the
    # original Kimi Linear 48B, which only ever ships g_a_proj/g_b_proj).
    output_t0 = profile.start_substep() if profile is not None else None
    # GLM-5.3's released recurrent/chunk kernels explicitly return the core
    # attention in the Q/K/V projection dtype while retaining the recurrent
    # endpoint in FP32.  The shared Kimi implementation historically kept the
    # core output FP32, so make this behavior explicit and opt-in at the GLM
    # call site instead of silently changing established Kimi profiles.
    if released_output_dtype:
        o = o.astype(projection_dtype)
    if cfg.kda_use_full_rank_gate:
        g_out = _linear(h, w, f"{prefix}.self_attn.g_proj")
    else:
        g_out = _linear(_linear(h, w, f"{prefix}.self_attn.g_a_proj"), w, f"{prefix}.self_attn.g_b_proj")
    g_out = g_out.reshape(B, L, H, D)
    o = _gated_rms_norm(o, g_out, w[f"{prefix}.self_attn.o_norm.weight"], cfg.rms_norm_eps)
    o = o.reshape(B, L, H * D)
    result = _linear(o, w, f"{prefix}.self_attn.o_proj")
    if profile is not None:
        profile.finish_substep(
            "kda_output", layer, output_t0, result, positions=L)
    return result


def _kimi_linear_attention_residual(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv,
    layer: int, offset: int, mlp_last_only: bool = False,
    native_fused_decode: bool = False,
) -> mx.array:
    """Attention (KDA or MLA) + residual only, no MLP/MoE. Split out of
    the original monolithic `run_kimi_linear_block` (F35-prep, 2026-07-24)
    so layer-stationary tiled prefill can call this PER TILE (attention
    must still see tiles in causal/sequential order -- KDA's recurrent
    state and MLA's KV cache both accumulate exactly as before, this split
    changes nothing about that) while calling the MLP/MoE half exactly
    ONCE per layer across the whole prompt instead. `run_kimi_linear_block`
    below is now a thin two-call wrapper preserving the exact original
    behavior for existing (chunk-major) callers.

    `kv` carries KDA's recurrent state the same way GLM's MLA carries
    `kv.compressed_mla`/`kv.dsa` -- an ad-hoc `kv.kda_cache` (KDAStateCache)
    attribute set once in Engine.new_kv(), not a separate threaded argument.
    A bare KVCache (or None, as the oracle/smoke tests pass) has no
    `kda_cache` attribute -- getattr defaults to a fresh-each-call None,
    i.e. stateless single-shot behavior, matching those tests' expectations.
    """
    h = mx.fast.rms_norm(x, w[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)

    if layer in cfg.full_attn_layers:
        x = x + _mla_attention(h, w, prefix, cfg, kv, layer, offset)
    elif layer in cfg.kda_layers:
        kda_cache = getattr(kv, "kda_cache", None)
        x = x + _kda_attention(
            h, w, prefix, cfg, kda_cache, layer,
            native_fused_decode=native_fused_decode)
    else:
        raise ValueError(
            f"layer {layer} is in neither cfg.full_attn_layers nor cfg.kda_layers")

    if mlp_last_only:  # KV/state is built; only the last position feeds the logits
        x = x[:, -1:, :]
    return x


def _kimi_linear_mlp_residual(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, layer: int,
    get_experts, iter_expert_batches=None, profile=None,
) -> mx.array:
    """MLP (dense) or MoE + residual only, given `x` already post-attention.
    See `_kimi_linear_attention_residual`'s docstring for why this is split
    out. `x` may cover any subset of positions (a single tile, or the
    whole prompt) -- routing/expert-fetch always operates on exactly
    whatever positions are present in `x`, which is what lets a
    layer-stationary caller route the WHOLE prompt at once instead of
    per-tile."""
    h = mx.fast.rms_norm(x, w[f"{prefix}.post_attention_layernorm.weight"], cfg.rms_norm_eps)

    if layer < cfg.first_k_dense_replace:
        return x + _kimi_dense_mlp(h, w, f"{prefix}.mlp", cfg)

    return x + _kimi_moe_output(h, w, prefix, cfg, layer, get_experts, iter_expert_batches, profile)


def _kimi_moe_output(
    h: mx.array, w: dict, prefix: str, cfg: ModelConfig, layer: int,
    get_experts, iter_expert_batches=None, profile=None, stat_collector=None,
    overlap_shared_expert: bool = False, shared_tile_size: int = 0,
) -> mx.array:
    """Routed-experts + shared-experts MoE output ONLY -- no residual add,
    and the caller must already have checked `layer >= cfg.first_k_dense_replace`
    (the dense layer-0 case is not handled here). Factored out of
    `_kimi_linear_mlp_residual` (which just does `x + this`) so the
    AttnRes-aware K3 block runner below can reuse the exact same MoE math
    without duplicating it -- AttnRes replaces what happens to the residual
    stream around each sublayer, not the sublayer's own computation.

    `stat_collector`, when given, is called as
    `stat_collector(layer, expert_id, route_weights, raw_expert_output)` for
    every routed expert actually executed this call -- `route_weights` is
    the router's gate weight per selected token (g_j(x)) and
    `raw_expert_output` is that expert's output BEFORE the gate multiply
    (e_j(x)), i.e. exactly the two REAP saliency-score ingredients
    (S_j = mean(g_j(x) * ||e_j(x)||_2), see experiments/
    kimi_k3_reap_calibrate.py). None (the default) adds zero overhead to
    the ordinary forward path -- calibration is opt-in instrumentation,
    never on by default."""
    moe_prefix = f"{prefix}.block_sparse_moe"
    router_t0 = profile.start_substep() if profile is not None else None
    idx, pw = _route_experts(h, w, moe_prefix, cfg, layer=layer)
    if not (profile is not None and profile.finish_substep(
            "router", layer, router_t0, idx, pw,
            positions=int(h.shape[1]))):
        mx.eval(idx, pw)
    groups = _group_routes(idx, pw)
    expert_ids = sorted(groups)
    positions_by_expert = {
        e: [pt for pt, _ in groups[e]] for e in expert_ids
    }
    # Routing is authoritative now. A pipelined engine submits exact batch
    # zero while the resident latent/shared branches below run; an ordinary
    # non-pipelined iterator remains lazy and preserves its prior schedule.
    batches = (
        iter_expert_batches(
            layer, expert_ids, positions=positions_by_expert
        )
        if iter_expert_batches is not None
        else None
    )

    # F128: Kimi K3's real KimiSparseMoeBlock routes on the FULL hidden
    # state (h, above) but runs each expert in a smaller "latent" space
    # (config.routed_expert_hidden_size) when cfg.moe_latent_hidden_size is
    # set -- confirmed by the real routed_expert_down_proj/_norm/_up_proj
    # tensors on a real downloaded shard. Kimi Linear/K2.5 leave this 0, so
    # h_latent is just h unchanged for them (identical behavior to before
    # this branch existed).
    if cfg.moe_latent_hidden_size:
        h_latent = _linear(h, w, f"{moe_prefix}.routed_expert_down_proj")
        if batches is not None:
            mx.async_eval(h_latent)
    else:
        h_latent = h

    # The routed and shared branches are independent functions of the same
    # immutable h:
    #
    #     MoE(h) = R(h) + S(h).
    #
    # Submit S(h) before the first routed-weight fetch so Metal can consume the
    # resident shared weights while storage materializes routed batch zero.
    # The final addition and every routed accumulation remain in their original
    # order, so this is a scheduling identity rather than a math approximation.
    shared = None
    if shared_tile_size and cfg.moe_latent_hidden_size:
        # Long-context K3 already retains a full hidden-width AttnRes MLP
        # input.  Materializing the shared expert's full gate/up activations
        # beside it added another multi-GB peak at 46K.  Both routed branches
        # need only the much smaller latent projection after routing, while
        # the shared branch is row-independent.  Evaluate both now, with the
        # shared MLP position-tiled, then release the hidden-width input before
        # expert paging.  The final addition and every per-row dot product are
        # unchanged; only overlap/lifetime scheduling differs.
        shared = _kimi_dense_mlp_tiled(
            h,
            w,
            f"{moe_prefix}.shared_experts",
            cfg,
            shared_tile_size,
        )
        mx.eval(h_latent, shared)
        del h
    elif overlap_shared_expert:
        shared = _kimi_dense_mlp(
            h, w, f"{moe_prefix}.shared_experts", cfg,
            synchronize_subprojections=(
                cfg.model_type == "kimi_k3" and h.shape[1] == 1),
        )
        mx.async_eval(shared)

    out = mx.zeros_like(h_latent)
    if batches is None:
        experts = get_experts(layer, expert_ids, positions=positions_by_expert)
        batches = ((expert_ids, experts),)

    def consume_batch(batch_ids, experts):
        nonlocal out
        for e in batch_ids:
            plist = groups[e]
            positions = [p for p, _ in plist]
            route_weights = mx.array([wt for _, wt in plist]).astype(mx.float32)
            y = _kimi_expert_mlp(h_latent[:, positions, :], experts[e], f"{moe_prefix}.experts.{e}", cfg)
            if stat_collector is not None:
                stat_collector(layer, e, route_weights, y)
            contribution = (y * route_weights[None, :, None]).astype(h_latent.dtype)
            out = out.at[:, positions, :].add(contribution)
        mx.eval(out)
        del contribution, y, route_weights

    consume_expert_batches(batches, consume_batch)

    if cfg.moe_latent_hidden_size:
        if cfg.moe_latent_use_norm:
            out = mx.fast.rms_norm(
                out, w[f"{moe_prefix}.routed_expert_norm.weight"], cfg.rms_norm_eps)
        mx.eval(out)
        del h_latent
        if shared is not None and shared_tile_size:
            # Both final branches are row-independent:
            #
            #   result[p] = up(routed[p]) + shared[p].
            #
            # The ordinary expression materializes a second full hidden-width
            # buffer for ``up(routed)`` and a third for the addition. Reuse the
            # already-evaluated shared output as the destination and evaluate
            # the identical latent-width dot product in position tiles. The
            # reduction dimension and left-to-right addition are unchanged;
            # only independent output rows and tensor lifetimes are scheduled.
            for start in range(0, out.shape[1], shared_tile_size):
                end = min(start + shared_tile_size, out.shape[1])
                routed_tile = _linear(
                    out[:, start:end, :],
                    w,
                    f"{moe_prefix}.routed_expert_up_proj",
                )
                value = routed_tile + shared[:, start:end, :]
                mx.eval(value)
                shared[:, start:end, :] = value
                mx.eval(shared)
            del out
            return shared
        out = _linear(out, w, f"{moe_prefix}.routed_expert_up_proj")

    if shared is None:
        shared = _kimi_dense_mlp(
            h, w, f"{moe_prefix}.shared_experts", cfg,
            synchronize_subprojections=(
                cfg.model_type == "kimi_k3" and h.shape[1] == 1),
        )
    else:
        mx.eval(shared)
    return out + shared


def run_kimi_linear_block(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv,
    layer: int, offset: int, get_experts, mlp_last_only: bool = False, iter_expert_batches=None,
    native_fused_decode: bool = False,
    profile=None,
) -> mx.array:
    """One Kimi Linear decoder block (chunk-major / ordinary use). Thin
    wrapper over `_kimi_linear_attention_residual` +
    `_kimi_linear_mlp_residual` -- see those functions' docstrings for why
    the split exists (layer-stationary tiled prefill, F35-prep)."""
    positions = int(x.shape[1])
    attention_t0 = profile.start_substep() if profile is not None else None
    x = _kimi_linear_attention_residual(
        x, w, prefix, cfg, kv, layer, offset, mlp_last_only=mlp_last_only,
        native_fused_decode=native_fused_decode)
    if profile is not None:
        profile.finish_substep(
            "attention", layer, attention_t0, x, positions=positions)
    mlp_t0 = profile.start_substep() if profile is not None else None
    x = _kimi_linear_mlp_residual(
        x, w, prefix, cfg, layer, get_experts,
        iter_expert_batches=iter_expert_batches, profile=profile)
    if profile is not None:
        profile.finish_substep(
            "mlp", layer, mlp_t0, x, positions=int(x.shape[1]))
    return x


_FUSED_ATTNRES_SOURCE = r"""
    constexpr float NEG_INF = -3.402823466e+38f;

    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint group = simdgroup_index_in_threadgroup;
    const int blocks = counts[0];
    const int sources = blocks + 1;
    const float epsilon = eps[0];

    threadgroup float partial_square[GROUPS];
    threadgroup float partial_score[GROUPS];
    threadgroup float logits[MAX_SOURCES];
    threadgroup float probabilities[MAX_SOURCES];

    // Moonshot/FLA's fused formulation: compute the RMS statistic and the
    // learned scalar logit while each source row is resident in the kernel.
    // A second source pass emits the weighted residual.  Keeping H/BCOUNT out
    // of global float32 temporaries is the important Metal-memory property;
    // caching every H-wide source in registers is not viable at K3 H=7168.
    for (int source = 0; source < sources; ++source) {
        float square = 0.0f;
        float score = 0.0f;
        for (uint dim = tid; dim < HIDDEN; dim += THREADS) {
            const size_t index = source < blocks
                ? ((size_t)row * blocks + source) * HIDDEN + dim
                : (size_t)row * HIDDEN + dim;
            const float value = source < blocks
                ? static_cast<float>(residual[index])
                : static_cast<float>(prefix[(size_t)row * HIDDEN + dim]);
            square += value * value;
            score += value
                * static_cast<float>(norm_weight[dim])
                * static_cast<float>(proj_weight[dim]);
        }
        square = simd_sum(square);
        score = simd_sum(score);
        if (lane == 0) {
            partial_square[group] = square;
            partial_score[group] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (group == 0) {
            const float square_part =
                lane < GROUPS ? partial_square[lane] : 0.0f;
            const float score_part =
                lane < GROUPS ? partial_score[lane] : 0.0f;
            const float square_sum = simd_sum(square_part);
            const float score_sum = simd_sum(score_part);
            if (lane == 0) {
                logits[source] = score_sum * rsqrt(
                    square_sum / static_cast<float>(HIDDEN) + epsilon
                );
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float maximum = NEG_INF;
        for (int source = 0; source < sources; ++source) {
            maximum = max(maximum, logits[source]);
        }
        float denominator = 0.0f;
        for (int source = 0; source < sources; ++source) {
            const float value = exp(logits[source] - maximum);
            probabilities[source] = value;
            denominator += value;
        }
        for (int source = 0; source < sources; ++source) {
            probabilities[source] /= denominator;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = tid; dim < HIDDEN; dim += THREADS) {
        float value = 0.0f;
        for (int source = 0; source < sources; ++source) {
            const size_t index = source < blocks
                ? ((size_t)row * blocks + source) * HIDDEN + dim
                : (size_t)row * HIDDEN + dim;
            const float source_value = source < blocks
                ? static_cast<float>(residual[index])
                : static_cast<float>(prefix[(size_t)row * HIDDEN + dim]);
            value += probabilities[source] * source_value;
        }
        out[(size_t)row * HIDDEN + dim] = static_cast<T>(value);
    }
"""


@lru_cache(maxsize=1)
def _fused_attnres_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="voom_kimi_k3_fused_attnres",
        input_names=[
            "prefix",
            "residual",
            "proj_weight",
            "norm_weight",
            "eps",
            "counts",
        ],
        output_names=["out"],
        source=_FUSED_ATTNRES_SOURCE,
    )


def _apply_attn_res_reference(
    prefix_sum: mx.array, block_residual,
    proj_weight: mx.array, norm_weight: mx.array, eps: float,
) -> mx.array:
    """F128: Kimi K3's "Attention Residuals" (AttnRes, arXiv 2603.15031),
    ported verbatim from the real modeling_kimi_linear.py's module-level
    `_apply_attn_res`: a softmax-attention readout over `block_residual`
    (residual-stream snapshots taken every `cfg.attn_res_block_size`
    layers, one column per snapshot so far) PLUS the current running
    `prefix_sum`, using a shared RMSNorm + single learned scalar projection
    per query/key ("proj"/"norm" are `nn.Linear(hidden,1,bias=False)` and
    an RMSNorm respectively -- NOT the usual QK attention shapes, there is
    only one score per snapshot, not a per-head/per-dim breakdown).

    prefix_sum: (N, hidden). block_residual: (N, num_blocks, hidden), where
    N = batch*positions and num_blocks may be 0 (no snapshot taken yet --
    callers must skip calling this entirely in that case, matching the real
    code's `if block_residual.shape[1] > 0` guard around its first call
    site only; every other call site always has num_blocks >= 1 by
    construction). Returns (N, hidden).
    """
    if isinstance(block_residual, (list, tuple)):
        block_residual = (
            mx.stack(block_residual, axis=1)
            if block_residual
            else mx.zeros(
                (prefix_sum.shape[0], 0, prefix_sum.shape[1]),
                dtype=prefix_sum.dtype,
            )
        )
    v = mx.concatenate([block_residual, prefix_sum[:, None, :]], axis=1)
    v32 = v.astype(mx.float32)
    variance = mx.mean(v32 * v32, axis=-1, keepdims=True)
    k = v32 * mx.rsqrt(variance + eps)
    score_weight = norm_weight.astype(mx.float32) * proj_weight.reshape(-1).astype(mx.float32)
    scores = mx.sum(k * score_weight, axis=-1)
    probs = mx.softmax(scores, axis=-1)[:, None, :]
    out = (probs @ v32)[:, 0, :]
    return out.astype(v.dtype)


def _apply_attn_res_fused_tiled(
    prefix_sum: mx.array,
    block_residual,
    proj_weight: mx.array,
    norm_weight: mx.array,
    eps: float,
    tile_size: int,
) -> mx.array:
    """Bounded-memory MLX/Metal port of FLA's fused AttnRes forward.

    One Metal threadgroup owns one position row and fuses RMS statistics,
    learned logits, stable softmax, and residual mixing.  Position tiling
    bounds command/lazy-graph lifetime and, crucially, never constructs the
    composite path's ``(tile, sources, hidden)`` float32 ``v/k`` tensors.
    """
    kernel = _fused_attnres_kernel()
    if kernel is None:
        return _apply_attn_res_reference(
            prefix_sum, block_residual, proj_weight, norm_weight, eps
        )
    if tile_size <= 0:
        raise ValueError("fused AttnRes tile_size must be positive")
    if prefix_sum.ndim != 2:
        raise ValueError("fused AttnRes expects a rank-2 prefix")
    rows, hidden = prefix_sum.shape
    residual_store = (
        block_residual
        if isinstance(block_residual, DiskBackedAttnResSnapshots)
        else None
    )
    residual_list = (
        list(block_residual)
        if residual_store is None
        and isinstance(block_residual, (list, tuple))
        else None
    )
    if residual_store is not None:
        if residual_store.shape != prefix_sum.shape:
            raise ValueError(
                "packed AttnRes residual shape does not match prefix")
        blocks = len(residual_store)
    elif residual_list is None:
        if block_residual.ndim != 3:
            raise ValueError("fused AttnRes expects rank-3 residuals")
        if (
            block_residual.shape[0] != rows
            or block_residual.shape[2] != hidden
        ):
            raise ValueError(
                "fused AttnRes residual shape does not match prefix"
            )
        blocks = int(block_residual.shape[1])
    else:
        for residual in residual_list:
            if residual.ndim != 2 or residual.shape != prefix_sum.shape:
                raise ValueError(
                    "fused AttnRes residual list does not match prefix"
                )
        blocks = len(residual_list)
    if blocks + 1 > 16:
        # K3 has far fewer residual sources, but keep arbitrary compatible
        # models correct instead of indexing beyond fixed threadgroup storage.
        return _apply_attn_res_reference(
            prefix_sum, block_residual, proj_weight, norm_weight, eps
        )
    threads = min(
        1024,
        max(32, ((int(hidden) + 31) // 32) * 32),
    )
    groups = threads // 32
    epsilon = mx.array([float(eps)], dtype=mx.float32)
    counts = mx.array([blocks], dtype=mx.int32)
    output = mx.zeros(prefix_sum.shape, dtype=prefix_sum.dtype)
    mx.eval(output)
    for start in range(0, rows, tile_size):
        end = min(start + tile_size, rows)
        prefix_tile = prefix_sum[start:end]
        if residual_store is not None:
            residual_tile = residual_store.read_stacked(start, end)
        elif residual_list is None:
            residual_tile = block_residual[start:end]
        elif residual_list:
            # FLA's list-input/pointer-table idea expressed with MLX views:
            # snapshots remain independent full-position buffers, while only
            # this bounded tile is stacked into one contiguous kernel operand.
            # This removes every full-context snapshot concatenation/copy.
            residual_tile = mx.stack(
                [value[start:end] for value in residual_list],
                axis=1,
            )
        else:
            residual_tile = mx.zeros(
                (end - start, 0, hidden), dtype=prefix_sum.dtype
            )
        value = kernel(
            inputs=[
                prefix_tile,
                residual_tile,
                proj_weight.reshape(-1),
                norm_weight.reshape(-1),
                epsilon,
                counts,
            ],
            template=[
                ("T", prefix_sum.dtype),
                ("HIDDEN", int(hidden)),
                ("THREADS", threads),
                ("GROUPS", groups),
                ("MAX_SOURCES", 16),
            ],
            grid=(threads, end - start, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(end - start, hidden)],
            output_dtypes=[prefix_sum.dtype],
        )[0]
        mx.eval(value)
        output[start:end] = value
        mx.eval(output)
    return output


def _apply_attn_res(
    prefix_sum: mx.array, block_residual,
    proj_weight: mx.array, norm_weight: mx.array, eps: float,
    *, fused_tile_size: int = 0,
) -> mx.array:
    if fused_tile_size:
        return _apply_attn_res_fused_tiled(
            prefix_sum,
            block_residual,
            proj_weight,
            norm_weight,
            eps,
            fused_tile_size,
        )
    return _apply_attn_res_reference(
        prefix_sum, block_residual, proj_weight, norm_weight, eps
    )


def attn_res_attention_input(
    x: mx.array, block_residual, w: dict, prefix: str,
    cfg: ModelConfig, layer: int, *, fused_tile_size: int = 0,
) -> tuple[mx.array | None, mx.array, mx.array]:
    """Apply AttnRes bookkeeping through the normalized attention input."""
    B, L, H = x.shape
    prefix_sum = x
    hidden_states = x

    block_count = (
        len(block_residual)
        if isinstance(block_residual, (list, tuple))
        else block_residual.shape[1]
    )
    if block_count > 0:
        hidden_states = _apply_attn_res(
            prefix_sum.reshape(-1, H), block_residual,
            w[f"{prefix}.self_attention_res_proj.weight"],
            w[f"{prefix}.self_attention_res_norm.weight"], cfg.rms_norm_eps,
            fused_tile_size=fused_tile_size,
        ).reshape(B, L, H)

    if layer % cfg.attn_res_block_size == 0:
        snapshot = prefix_sum.reshape(-1, H)
        if isinstance(block_residual, list):
            block_residual.append(snapshot)
        elif isinstance(block_residual, tuple):
            block_residual = [*block_residual, snapshot]
        else:
            block_residual = mx.concatenate(
                [block_residual, snapshot[:, None, :]], axis=1)
        prefix_sum = None

    attention_input = mx.fast.rms_norm(
        hidden_states,
        w[f"{prefix}.input_layernorm.weight"],
        cfg.rms_norm_eps,
    )
    return prefix_sum, block_residual, attention_input


def attn_res_mlp_input(
    prefix_sum: mx.array, block_residual, w: dict,
    prefix: str, cfg: ModelConfig, *, fused_tile_size: int = 0,
) -> mx.array:
    """Apply the post-attention AttnRes read and MLP normalization."""
    B, L, H = prefix_sum.shape
    hidden_states = _apply_attn_res(
        prefix_sum.reshape(-1, H), block_residual,
        w[f"{prefix}.mlp_res_proj.weight"],
        w[f"{prefix}.mlp_res_norm.weight"], cfg.rms_norm_eps,
        fused_tile_size=fused_tile_size,
    ).reshape(B, L, H)
    return mx.fast.rms_norm(
        hidden_states,
        w[f"{prefix}.post_attention_layernorm.weight"],
        cfg.rms_norm_eps,
    )


def attn_res_wrap_layer(
    x: mx.array, block_residual, w: dict, prefix: str,
    cfg: ModelConfig, layer: int, attn_fn, mlp_fn,
    *, fused_tile_size: int = 0,
) -> tuple[mx.array, mx.array]:
    """The AttnRes bookkeeping itself, ported verbatim from the real
    `KimiDecoderLayer._forward_attn_residual`'s control flow, factored out
    from any particular attention/MLP math so it can be unit-tested against
    a real-reference torch transcription with trivial stand-in
    `attn_fn`/`mlp_fn` (see tests/test_f128_k3_attn_res_oracle.py) --
    KDA/MLA/MoE math is already independently oracle-verified elsewhere
    (F92/F93), so this isolates the genuinely new risk: getting the
    block-boundary reset/snapshot bookkeeping itself right.

    `attn_fn`/`mlp_fn` each take the appropriately-normed hidden state and
    return the sublayer's raw output (pre-residual) -- `run_kimi_k3_block`
    below supplies the real KDA/MLA/dense/MoE closures; the oracle test
    supplies simple deterministic stand-ins instead.

    Returns `(new_prefix_sum, new_block_residual)`, both to be threaded
    into the next layer's call exactly like `x` itself already is.
    """
    prefix_sum, block_residual, attention_input = (
        attn_res_attention_input(
            x, block_residual, w, prefix, cfg, layer,
            fused_tile_size=fused_tile_size,
        )
    )
    attn_out = attn_fn(attention_input)
    prefix_sum = (prefix_sum + attn_out) if prefix_sum is not None else attn_out

    mlp_out = mlp_fn(
        attn_res_mlp_input(
            prefix_sum, block_residual, w, prefix, cfg,
            fused_tile_size=fused_tile_size,
        )
    )
    prefix_sum = prefix_sum + mlp_out
    return prefix_sum, block_residual


def attn_res_wrap_layer_streamed(
    x: mx.array,
    block_residual: DiskBackedAttnResSnapshots,
    w: dict,
    prefix: str,
    cfg: ModelConfig,
    layer: int,
    attn_tile_fn,
    mlp_fn,
    *,
    tile_size: int,
    fused_tile_size: int,
) -> tuple[mx.array, DiskBackedAttnResSnapshots]:
    """Bound full-context K3 activation lifetimes around exact row tiles.

    A 46K x 7168 BF16 activation is about 661 MB.  The ordinary wrapper can
    retain the full normalized attention input, every attention result tile,
    their concatenation, and the full MLP input at the same time.  Snapshot
    spilling alone therefore cannot keep a 16-GB Mac below its Metal ceiling.

    AttnRes and RMSNorm are independent across position rows.  Evaluate those
    operations around each already-causal attention tile and assign its result
    into one full output buffer.  The MLP still receives the entire prompt, so
    router grouping/expert reuse and released arithmetic are unchanged.  Only
    tensor lifetime and assignment scheduling differ.
    """
    if tile_size <= 0 or fused_tile_size <= 0:
        raise ValueError("streamed AttnRes requires positive tile sizes")
    if not isinstance(block_residual, DiskBackedAttnResSnapshots):
        raise TypeError("streamed AttnRes requires disk-backed snapshots")
    batch, total, hidden = map(int, x.shape)
    old_residual_count = len(block_residual)
    boundary = layer % cfg.attn_res_block_size == 0
    if boundary:
        block_residual.append(x.reshape(-1, hidden))

    def residual_position_tile(count: int, start: int, end: int):
        parts = [
            block_residual.read_stacked(
                b * total + start,
                b * total + end,
                count=count,
            )
            for b in range(batch)
        ]
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)

    def apply_tile(
        prefix_tile: mx.array,
        residual_count: int,
        proj_weight: mx.array,
        norm_weight: mx.array,
        start: int,
        end: int,
    ) -> mx.array:
        residual_tiles = residual_position_tile(
            residual_count, start, end)
        return _apply_attn_res(
            prefix_tile.reshape(-1, hidden),
            residual_tiles,
            proj_weight,
            norm_weight,
            cfg.rms_norm_eps,
            fused_tile_size=fused_tile_size,
        ).reshape(batch, end - start, hidden)

    for start in range(0, total, tile_size):
        end = min(start + tile_size, total)
        source_tile = x[:, start:end, :]
        hidden_tile = source_tile
        if old_residual_count:
            hidden_tile = apply_tile(
                source_tile,
                old_residual_count,
                w[f"{prefix}.self_attention_res_proj.weight"],
                w[f"{prefix}.self_attention_res_norm.weight"],
                start,
                end,
            )
        attention_input = mx.fast.rms_norm(
            hidden_tile,
            w[f"{prefix}.input_layernorm.weight"],
            cfg.rms_norm_eps,
        )
        attention_out = attn_tile_fn(attention_input, start, end)
        value = attention_out if boundary else source_tile + attention_out
        mx.eval(value)
        # This function owns ``x`` exclusively. The exact post-attention tile
        # is fully evaluated before assignment, so its source view is no
        # longer needed and the 661-MB full-context input can double as the
        # output accumulator instead of coexisting with a second buffer.
        x[:, start:end, :] = value
        mx.eval(x)
    del (
        attention_input,
        attention_out,
        hidden_tile,
        old_residual_count,
        source_tile,
        value,
    )
    attention_sum = x
    del x

    mlp_input = mx.zeros_like(attention_sum)
    mx.eval(mlp_input)
    current_residual_count = len(block_residual)
    for start in range(0, total, tile_size):
        end = min(start + tile_size, total)
        hidden_tile = apply_tile(
            attention_sum[:, start:end, :],
            current_residual_count,
            w[f"{prefix}.mlp_res_proj.weight"],
            w[f"{prefix}.mlp_res_norm.weight"],
            start,
            end,
        )
        value = mx.fast.rms_norm(
            hidden_tile,
            w[f"{prefix}.post_attention_layernorm.weight"],
            cfg.rms_norm_eps,
        )
        mx.eval(value)
        mlp_input[:, start:end, :] = value
        mx.eval(mlp_input)
    del current_residual_count

    # Pop the sole owner before entering the callback.  The K3 latent-MoE
    # callback can then release its hidden-width input as soon as routing,
    # latent projection, and the tiled shared branch have consumed it.
    mlp_input_owner = [mlp_input]
    del mlp_input
    mlp_out = mlp_fn(mlp_input_owner.pop())
    mx.eval(mlp_out)
    # The residual addition is also independent per position. Reuse the
    # post-attention accumulator after each exact tile result is materialized,
    # avoiding a third full hidden-width buffer at the layer boundary.
    for start in range(0, total, tile_size):
        end = min(start + tile_size, total)
        value = attention_sum[:, start:end, :] + mlp_out[:, start:end, :]
        mx.eval(value)
        attention_sum[:, start:end, :] = value
        mx.eval(attention_sum)
    del mlp_out, value
    return attention_sum, block_residual


def run_kimi_k3_block(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv,
    layer: int, offset: int, block_residual, get_experts,
    mlp_last_only: bool = False, iter_expert_batches=None,
    native_fused_decode: bool = False,
    native_fused_prefill: bool = False,
    compiled_prefill: bool = False,
    profile=None, stat_collector=None,
    fused_attnres_tile_size: int = 0,
) -> tuple[mx.array, mx.array]:
    """One Kimi K3 decoder block WITH AttnRes -- thin wrapper supplying the
    real KDA/MLA attention and dense/MoE MLP closures to
    `attn_res_wrap_layer` above (see its docstring for the AttnRes mechanism
    itself). Unlike `run_kimi_linear_block` (whose `x = x + sublayer_out`
    residual this does NOT use), the running accumulator is `prefix_sum`:
    it resets to just the sublayer's own output at every block boundary
    instead of adding onto the prior value. `block_residual` is this
    function's extra piece of state, threaded by the caller exactly like
    `x` itself -- purely an intra-forward-pass, depth-wise accumulator (the
    real reference re-inits it fresh at the top of every `forward()` call
    and never persists it via any KV/cache mechanism), so chunk-major
    callers must start a fresh empty `block_residual` at the top of each
    chunk's layer loop, never carrying it across chunks or decode steps.
    Returns `(x, block_residual)`, both to be threaded into the next
    layer's call.
    """
    def attn_fn(h):
        if layer in cfg.full_attn_layers:
            return _mla_attention(h, w, prefix, cfg, kv, layer, offset)
        if layer in cfg.kda_layers:
            kda_cache = getattr(kv, "kda_cache", None)
            return _kda_attention(
                h, w, prefix, cfg, kda_cache, layer,
                native_fused_decode=native_fused_decode,
                native_fused_prefill=native_fused_prefill,
                compiled_prefill=compiled_prefill,
                profile=profile)
        raise ValueError(
            f"layer {layer} is in neither cfg.full_attn_layers nor cfg.kda_layers")

    def mlp_fn(h2):
        if layer < cfg.first_k_dense_replace:
            return _kimi_dense_mlp(
                h2, w, f"{prefix}.mlp", cfg,
                synchronize_subprojections=(h2.shape[1] == 1),
            )
        return _kimi_moe_output(
            h2, w, prefix, cfg, layer, get_experts,
            iter_expert_batches=iter_expert_batches, profile=profile,
            stat_collector=stat_collector)

    # F128: unlike run_kimi_linear_block's mlp_last_only (which trims BETWEEN
    # attention and MLP to skip MLP compute for positions whose logits are
    # never needed), this function deliberately ignores `mlp_last_only` and
    # always processes the full L positions through both attention and MLP.
    # Trimming here would shrink x's row count out from under block_residual
    # (built up over ALL positions across every earlier layer in this same
    # sweep) with no matching trim on block_residual's own rows, breaking
    # the row-alignment attn_res_wrap_layer's concatenation depends on. The
    # caller (Engine._sweep) trims AFTER the whole layer loop AND
    # apply_output_attn_res have both run instead -- see its own comment.
    # `mlp_last_only` is accepted only so this function's call signature
    # matches run_kimi_linear_block's; wasted MLP compute on discarded
    # positions is a real, un-optimized cost here, not a correctness issue.
    del mlp_last_only
    return attn_res_wrap_layer(
        x, block_residual, w, prefix, cfg, layer, attn_fn, mlp_fn,
        fused_tile_size=fused_attnres_tile_size)


def apply_output_attn_res(
    x: mx.array, w: dict, block_residual, cfg: ModelConfig,
    *, fused_tile_size: int = 0,
) -> mx.array:
    """The final AttnRes readout applied once after ALL layers (real
    `KimiLinearModel._apply_output_attn_res`), before the model's final
    RMSNorm -- uses its own dedicated `model.output_attn_res_proj`/
    `model.output_attn_res_norm` weights, distinct from any per-layer ones."""
    B, L, H = x.shape
    return _apply_attn_res(
        x.reshape(-1, H), block_residual,
        w["model.output_attn_res_proj.weight"],
        w["model.output_attn_res_norm.weight"], cfg.rms_norm_eps,
        fused_tile_size=fused_tile_size,
    ).reshape(B, L, H)
