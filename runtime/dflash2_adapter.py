"""Default-off Qwen3.8 DFlash2 adapter over the DSpark target verifier.

The architecture is adapted from z-lab/dflash's ``dflash/model_mlx.py`` at
revision ``07ebd93db9f472af339b644bb70221ad8428328a`` (Copyright (c) 2026
Z Lab, MIT).  The complete MIT notice is retained in :mod:`runtime.dflash2`.
This module is vOOM-specific glue: exact target compatibility, target residual
taps, bounded sliding draft context, sparse proposal-q expansion, and reuse of
DSpark's target-authoritative Qwen KV/DeltaNet commit path.

Server selection remains an explicit environment/profile action, and the
built sidecar continues to declare ``enabled_by_default=false``.  Promotion
still requires the real long forced-rejection recurrent-state oracle.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .dflash2 import (
    CandidateSelector,
    GroupedDynamicCausalConv,
)
from .dflash2_ablation import load_artifact as load_ablation_artifact
from .dflash2_schema import (
    DFlash2Config,
    OFFICIAL_REVISION,
    OFFICIAL_WEIGHTS_SHA256,
    _read_safetensors_header,
    sha256_file,
)
from .dflash2_sidecar import MANIFEST_NAME, MANIFEST_SCHEMA, SIDECAR_SCHEMA
from .dspark import (
    CtxCache,
    DSparkAttention,
    DSparkMLP,
    DSparkStats,
    DSparkSpeculativeDecoder,
)
from .kv_cache import KVCache
from .sampler import SamplingParams, sample_probabilities


MAX_QUANTIZED_BLOCK_SIZE = 5
MAX_QUANTIZED_PROPOSALS = MAX_QUANTIZED_BLOCK_SIZE - 1


@dataclass(frozen=True)
class DFlash2RuntimeConfig:
    checkpoint: DFlash2Config
    model_type: str = "qwen3_dflash2"
    target_model_type: str = "qwen3_5"
    share_target_embed: bool = True
    share_target_lm_head: bool = True
    logits_start: int = 1
    fused_dynamic_conv: bool = False
    ablation_enabled: bool = False
    ablation_strength: float = 0.0
    ablation_fingerprint: str = ""

    @property
    def hidden_size(self):
        return self.checkpoint.hidden_size

    @property
    def intermediate_size(self):
        return self.checkpoint.intermediate_size

    @property
    def vocab_size(self):
        return self.checkpoint.vocab_size

    @property
    def num_hidden_layers(self):
        return self.checkpoint.num_hidden_layers

    @property
    def num_attention_heads(self):
        return self.checkpoint.num_attention_heads

    @property
    def num_key_value_heads(self):
        return self.checkpoint.num_key_value_heads

    @property
    def head_dim(self):
        return self.checkpoint.head_dim

    @property
    def rms_norm_eps(self):
        return self.checkpoint.rms_norm_eps

    @property
    def rope_theta(self):
        return self.checkpoint.rope_theta

    @property
    def rope_parameters(self):
        return {
            "rope_theta": self.checkpoint.rope_theta,
            "rope_type": self.checkpoint.rope_type,
        }

    @property
    def attention_bias(self):
        return False

    @property
    def target_hidden_size(self):
        return self.checkpoint.hidden_size

    @property
    def block_size(self):
        return self.checkpoint.block_size

    @property
    def mask_token_id(self):
        return self.checkpoint.mask_token_id

    @property
    def target_layer_ids(self):
        return list(self.checkpoint.target_layer_ids)

    @property
    def sliding_window(self):
        return self.checkpoint.sliding_window

    @property
    def conv_kernel_size(self):
        return self.checkpoint.conv_kernel_size

    @property
    def conv_group_size(self):
        return self.checkpoint.conv_group_size

    @property
    def selector_rank(self):
        return self.checkpoint.selector_rank

    @property
    def selector_top_k(self):
        return self.checkpoint.selector_top_k


def _target_value(target_config, name: str, default=None):
    return getattr(target_config, name, default)


def validate_target_compatibility(
    config: DFlash2RuntimeConfig,
    target_config,
) -> None:
    """Fail closed on target architecture, not target weight identity.

    Huihui's abliterated target intentionally differs from the official target
    used to train the drafter; exact verification permits that difference.
    Tensor geometry, tokenizer vocabulary, RoPE, and hybrid layer topology are
    not interchangeable and must still match.
    """
    checkpoint = config.checkpoint
    expected = {
        "model_type": "qwen3_5",
        "hidden_size": checkpoint.hidden_size,
        "intermediate_size": checkpoint.intermediate_size,
        "vocab_size": checkpoint.vocab_size,
        "num_hidden_layers": checkpoint.num_target_layers,
        "rope_theta": checkpoint.rope_theta,
        "max_position_embeddings": checkpoint.max_position_embeddings,
        "tie_word_embeddings": False,
        "attention_bias": False,
    }
    mismatches = []
    for name, value in expected.items():
        actual = _target_value(target_config, name)
        if actual != value:
            mismatches.append(f"{name}={actual!r} (expected {value!r})")

    layer_types = tuple(_target_value(target_config, "layer_types", ()))
    expected_layers = tuple(
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(checkpoint.num_target_layers))
    if layer_types != expected_layers:
        mismatches.append("layer_types do not match Qwen3.8's 3:1 hybrid layout")

    # These are target-side Qwen3.8 shapes, intentionally distinct from the
    # smaller attention geometry inside the draft checkpoint.
    qwen38_target = {
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    if checkpoint.num_target_layers == 64 and checkpoint.hidden_size == 5120:
        for name, value in qwen38_target.items():
            actual = _target_value(target_config, name)
            if actual != value:
                mismatches.append(f"{name}={actual!r} (expected {value!r})")
    if mismatches:
        raise ValueError(
            "DFlash2/target compatibility failure: " + "; ".join(mismatches))


class DFlash2Attention(DSparkAttention):
    """DFlash cross-attention with the published noncausal sliding mask."""

    def __init__(self, config: DFlash2RuntimeConfig):
        super().__init__(config)
        self.sliding_window = config.sliding_window

    def attend(
        self, hidden: mx.array, block_offset: int, cache: CtxCache,
    ) -> mx.array:
        batch, query_length, _ = hidden.shape
        queries = self.q_proj(hidden).reshape(
            batch, query_length, self.n_heads, self.head_dim)
        queries = self._rope(
            self.q_norm(queries).transpose(0, 2, 1, 3),
            offset=block_offset,
        )
        block_keys, block_values = self._kv(hidden)
        block_keys = self._rope(block_keys, offset=block_offset)
        keys = (
            mx.concatenate([cache.k, block_keys], axis=2)
            if cache.k is not None else block_keys)
        values = (
            mx.concatenate([cache.v, block_values], axis=2)
            if cache.v is not None else block_values)
        context_length = 0 if cache.k is None else int(cache.k.shape[2])

        mask = dflash2_sliding_mask(
            context_length, query_length, self.sliding_window)
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(
            batch, query_length, -1)
        return self.o_proj(output)


def dflash2_sliding_mask(
    context_length: int,
    query_length: int,
    sliding_window: int,
) -> mx.array:
    """Published noncausal-block/sliding-context DFlash attention mask."""
    if min(context_length, query_length) < 0 or sliding_window <= 0:
        raise ValueError("DFlash2 attention mask dimensions are invalid")
    query = context_length + mx.arange(query_length)[:, None]
    key = mx.arange(context_length + query_length)[None]
    context = mx.logical_and(
        key < context_length,
        query - key < sliding_window,
    )
    # is_causal=false is architectural: every draft position can attend every
    # position in the same noisy proposal block, including later mask slots.
    block = key >= context_length
    return mx.logical_or(context, block)


class DFlash2DecoderLayer(nn.Module):
    def __init__(self, config: DFlash2RuntimeConfig):
        super().__init__()
        self.self_attn = DFlash2Attention(config)
        self.mlp = DSparkMLP(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size,
            config.conv_kernel_size,
            config.conv_group_size,
            fused=config.fused_dynamic_conv,
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            config.hidden_size,
            config.conv_kernel_size,
            config.conv_group_size,
            fused=config.fused_dynamic_conv,
        )

    def __call__(
        self,
        hidden: mx.array,
        block_offset: int,
        cache: CtxCache,
        *,
        residual_direction: mx.array | None = None,
        residual_strength: float = 1.0,
        project_residual=None,
    ) -> mx.array:
        if residual_direction is not None and project_residual is not None:
            raise ValueError("DFlash2 residual projection was specified twice")
        residual = hidden
        hidden, kernel = self.attention_conv.prepare(
            self.input_layernorm(hidden))
        branch = self.attention_conv.finish(
            self.self_attn.attend(hidden, block_offset, cache),
            kernel,
            projection_direction=residual_direction,
            projection_strength=residual_strength,
        )
        if project_residual is not None:
            branch = project_residual(branch)
        hidden = residual + branch
        residual = hidden
        hidden, kernel = self.mlp_conv.prepare(
            self.post_attention_layernorm(hidden))
        branch = self.mlp_conv.finish(
            self.mlp(hidden),
            kernel,
            projection_direction=residual_direction,
            projection_strength=residual_strength,
        )
        if project_residual is not None:
            branch = project_residual(branch)
        return residual + branch


def expand_sparse_candidate_probabilities(
    candidates: mx.array,
    probabilities: mx.array,
    vocab_size: int,
) -> list[mx.array]:
    """Expand conditional top-k selector q rows for DSpark's exact verifier."""
    if candidates.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("DFlash2 sparse q inputs must be rank 2")
    if candidates.shape != probabilities.shape:
        raise ValueError("DFlash2 candidate IDs and q shapes differ")
    if vocab_size <= 0:
        raise ValueError("DFlash2 vocabulary size must be positive")
    rows = []
    for position in range(candidates.shape[0]):
        row = mx.zeros((vocab_size,), dtype=mx.float32).at[
            candidates[position]
        ].add(probabilities[position].astype(mx.float32))
        rows.append(row)
    return rows


def build_proposal_block(
    pending_token: int,
    mask_token_id: int,
    checkpoint_block_size: int,
    proposal_count: int,
) -> list[int]:
    """Build the exact Q4 runtime block (anchor plus at most four masks)."""
    if proposal_count <= 0 or proposal_count > MAX_QUANTIZED_PROPOSALS:
        raise ValueError(
            f"DFlash2 quantized proposal count must be in [1, "
            f"{MAX_QUANTIZED_PROPOSALS}]")
    if proposal_count >= checkpoint_block_size:
        raise ValueError("DFlash2 proposal count exceeds checkpoint block")
    return [int(pending_token)] + [int(mask_token_id)] * proposal_count


def greedy_candidate_recall(
    emitted: list[int],
    rounds: list[tuple[int, int]],
    candidate_rounds: list[list[list[int]]],
) -> tuple[int, int]:
    """Return target-decision count/hits inside the draft top-k support.

    Only the accepted prefix and first mismatch affect greedy progress and can
    be reconstructed from the committed stream. Rejected-tail predictions are
    intentionally excluded instead of being guessed from later output.
    """
    positions = hits = 0
    cursor = 1
    for (proposed, accepted), candidates in zip(
        rounds, candidate_rounds, strict=False,
    ):
        decisions = min(int(proposed), int(accepted) + 1)
        for position in range(min(decisions, len(candidates))):
            token_index = cursor + position
            if token_index >= len(emitted):
                break
            positions += 1
            hits += int(int(emitted[token_index]) in candidates[position])
        cursor += int(accepted) + 1
    return positions, hits


def _sidecar_physical_names(
    config: DFlash2Config,
    *,
    bits: int,
    group_size: int,
    mode: str,
) -> set[str]:
    names = set()
    for source_name, spec in config.expected_tensor_specs().items():
        output_name = (
            f"{source_name}.weight"
            if source_name in {
                "candidate_selector.predecessor_codebook",
                "candidate_selector.successor_codebook",
            }
            else source_name)
        if len(spec.shape) != 2 or spec.shape[-1] % group_size:
            names.add(output_name)
            continue
        names.add(output_name)
        stem = output_name[:-len(".weight")]
        names.add(f"{stem}.scales")
        if mode == "affine":
            names.add(f"{stem}.biases")
    return names


def inspect_runtime_sidecar(
    model_dir: str | Path,
    *,
    verify_hash: bool = True,
    require_official_geometry: bool = True,
) -> tuple[DFlash2RuntimeConfig, set[str], dict[str, Any]]:
    model_dir = Path(model_dir).resolve()
    raw = json.loads((model_dir / "config.json").read_text())
    checkpoint = DFlash2Config.from_mapping(raw)
    if require_official_geometry:
        checkpoint.validate_official_qwen38()
    sidecar = raw.get("vmodel_sidecar")
    if not isinstance(sidecar, dict):
        raise ValueError("DFlash2 artifact has no vmodel_sidecar identity")
    required_sidecar = {
        "schema": SIDECAR_SCHEMA,
        "draft_only_quantization": True,
        "target_verifier_required": True,
        "recurrent_rollback_oracle_required": True,
        "runtime_supported": False,
        "enabled_by_default": False,
    }
    if any(
        sidecar.get(name) != value for name, value in required_sidecar.items()
    ):
        raise ValueError("DFlash2 sidecar identity/default-off contract mismatch")
    if require_official_geometry and (
        sidecar.get("source_revision") != OFFICIAL_REVISION
        or sidecar.get("source_sha256") != OFFICIAL_WEIGHTS_SHA256
    ):
        raise ValueError("DFlash2 sidecar official source identity mismatch")
    manifest = json.loads((model_dir / MANIFEST_NAME).read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("DFlash2 sidecar manifest schema mismatch")
    conversion = manifest.get("conversion")
    if not isinstance(conversion, dict):
        raise ValueError("DFlash2 sidecar manifest omits conversion identity")
    quantization = raw.get("quantization")
    expected_quantization = {
        name: conversion.get(name) for name in ("bits", "group_size", "mode")}
    if quantization != expected_quantization:
        raise ValueError("DFlash2 config/manifest quantization mismatch")
    bits = int(quantization["bits"])
    group_size = int(quantization["group_size"])
    mode = str(quantization["mode"])
    if mode != "affine" or group_size != 64 or bits not in (2, 3, 4):
        raise ValueError(
            "DFlash2 runtime requires a pinned affine2/3/4 group64 sidecar")

    proof = manifest.get("proof")
    serving = manifest.get("serving")
    if not isinstance(proof, dict) or not isinstance(serving, dict):
        raise ValueError("DFlash2 sidecar manifest omits proof/serving gates")
    required_proof = {
        "source_full_sha256_verified": True,
        "source_header_validated": True,
        "target_weights_copied": False,
        "recurrent_rollback_proven": False,
        "runtime_supported": False,
    }
    required_serving = {
        "enabled_by_default": False,
        "runtime_supported": False,
        "target_verifier_required": True,
        "recurrent_rollback_oracle_required": True,
        "planned_proposal_count": (
            min(checkpoint.block_size, MAX_QUANTIZED_BLOCK_SIZE) - 1),
        "upstream_mlx_recommended_block_size": min(
            checkpoint.block_size, MAX_QUANTIZED_BLOCK_SIZE),
    }
    if any(proof.get(name) != value for name, value in required_proof.items()):
        raise ValueError("DFlash2 sidecar proof gate mismatch")
    if any(serving.get(name) != value
           for name, value in required_serving.items()):
        raise ValueError("DFlash2 sidecar default-off serving gate mismatch")

    weights_path = model_dir / "model.safetensors"
    header, _header_bytes, file_bytes = _read_safetensors_header(weights_path)
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("DFlash2 sidecar safetensors metadata is invalid")
    expected_metadata = {
        "format": "mlx",
        "vmodel_kind": "target-verified-dflash2-draft-sidecar",
        "runtime_supported": "false",
        "source_sha256": sidecar["source_sha256"],
        "source_revision": sidecar["source_revision"],
    }
    if any(metadata.get(name) != value
           for name, value in expected_metadata.items()):
        raise ValueError("DFlash2 safetensors identity mismatch")
    physical_names = set(header)
    expected_names = _sidecar_physical_names(
        checkpoint, bits=bits, group_size=group_size, mode=mode)
    if physical_names != expected_names:
        raise ValueError(
            "DFlash2 sidecar tensor set mismatch: "
            f"missing={sorted(expected_names - physical_names)[:8]} "
            f"unexpected={sorted(physical_names - expected_names)[:8]}")
    output = manifest.get("output")
    if not isinstance(output, dict) or output.get("weights_bytes") != file_bytes:
        raise ValueError("DFlash2 sidecar byte size differs from manifest")
    if verify_hash and sha256_file(weights_path) != output.get("weights_sha256"):
        raise ValueError("DFlash2 sidecar SHA-256 differs from manifest")
    return DFlash2RuntimeConfig(checkpoint), physical_names, manifest


class DFlash2Drafter(nn.Module):
    """Standalone draft model; the target owns embedding and LM head."""

    def __init__(self, config: DFlash2RuntimeConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            DFlash2DecoderLayer(config)
            for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.candidate_selector = CandidateSelector(
            config.hidden_size,
            config.vocab_size,
            config.selector_rank,
            config.selector_top_k,
        )
        self.confidence_head = None
        self._target_embed_provider = None
        self._target_lm_head_provider = None
        # Host-only proposal diagnostics. Candidate IDs are copied only after
        # the selector tensors have already been evaluated for generation, so
        # this adds no target sweep or additional model projection.
        self._last_candidate_ids: list[list[int]] = []
        self._last_unary_tokens: list[int] = []
        self._ablation_direction: mx.array | None = None

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        *,
        verify_hash: bool = True,
        require_official_geometry: bool = True,
        fused_dynamic_conv: bool = False,
    ) -> "DFlash2Drafter":
        config, physical_names, _manifest = inspect_runtime_sidecar(
            model_dir,
            verify_hash=verify_hash,
            require_official_geometry=require_official_geometry,
        )
        config = replace(
            config, fused_dynamic_conv=bool(fused_dynamic_conv))
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        quantization = raw["quantization"]
        model = cls(config)
        nn.quantize(
            model,
            group_size=int(quantization["group_size"]),
            bits=int(quantization["bits"]),
            mode=quantization.get("mode", "affine"),
            class_predicate=lambda module_path, _module: (
                f"{module_path}.scales" in physical_names),
        )
        model.load_weights(str(Path(model_dir) / "model.safetensors"))
        mx.eval(model.parameters())
        return model

    def set_ablation(
        self,
        direction: mx.array,
        *,
        strength: float,
        fingerprint: str,
    ) -> None:
        if direction.ndim != 1 or direction.shape[0] != self.config.hidden_size:
            raise ValueError("DFlash2 ablation direction width mismatch")
        if not 0.0 < float(strength) <= 2.0:
            raise ValueError("DFlash2 ablation strength must be in (0, 2]")
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("DFlash2 ablation fingerprint must be SHA-256")
        direction = direction.astype(mx.float32)
        mx.eval(direction)
        norm_squared = float(mx.sum(direction * direction).item())
        if not math.isfinite(norm_squared) or not math.isclose(
            norm_squared, 1.0, rel_tol=0, abs_tol=2e-5,
        ):
            raise ValueError("DFlash2 ablation direction must be unit-normalized")
        self._ablation_direction = direction
        self.config = replace(
            self.config,
            ablation_enabled=True,
            ablation_strength=float(strength),
            ablation_fingerprint=fingerprint,
        )

    def bind_target_embed(self, provider) -> None:
        self._target_embed_provider = provider

    def bind_target_lm_head(self, provider) -> None:
        self._target_lm_head_provider = provider

    def make_ctx_cache(self) -> list[CtxCache]:
        return [CtxCache() for _ in self.layers]

    def _embed_block(self, token_ids: list[int]) -> mx.array:
        if self._target_embed_provider is None:
            raise RuntimeError("DFlash2 requires the target token embedding")
        value = self._target_embed_provider(token_ids)
        if value.ndim != 3 or value.shape[:2] != (1, len(token_ids)):
            raise RuntimeError(
                "DFlash2 target embedding provider returned an invalid shape")
        return value

    def _project_logits(self, hidden: mx.array) -> mx.array:
        if self._target_lm_head_provider is None:
            raise RuntimeError("DFlash2 requires the target LM head")
        head = self._target_lm_head_provider()
        from .lm_head_stream import StreamedLMHead

        if isinstance(head, StreamedLMHead):
            return head.logits(hidden)
        from .layer_runner import _linear

        return _linear(
            hidden, {"dflash2_target_head.weight": head},
            "dflash2_target_head")

    @staticmethod
    def _trim_context_left(cache: CtxCache, max_length: int) -> None:
        drop = max(0, cache.length - max_length)
        if drop <= 0:
            return
        cache.k = cache.k[:, :, drop:, :]
        cache.v = cache.v[:, :, drop:, :]
        cache.position_start = int(cache.position_start or 0) + drop

    def update_context(
        self,
        target_hidden_cat: mx.array,
        ctx_offset: int,
        ctx_caches: list[CtxCache],
    ) -> None:
        fused = self.hidden_norm(self.fc(target_hidden_cat))
        max_context = self.config.sliding_window - 1
        for layer, cache in zip(self.layers, ctx_caches, strict=True):
            layer.self_attn.update_ctx(fused, ctx_offset, cache)
            self._trim_context_left(cache, max_context)

    def propose_block(
        self,
        pending_token: int,
        block_offset: int,
        ctx_caches: list[CtxCache],
        cap: int,
        sampling: SamplingParams,
        proposal_policy: str = "selector",
    ) -> tuple[list[int], list[mx.array] | None, mx.array]:
        if proposal_policy not in {"selector", "unary"}:
            raise ValueError("DFlash2 proposal_policy must be selector or unary")
        if proposal_policy == "unary" and not sampling.is_greedy:
            raise ValueError(
                "DFlash2 unary proposal policy is greedy-only")
        block_ids = build_proposal_block(
            pending_token, self.mask_token_id, self.config.block_size, cap)
        hidden = self._embed_block(block_ids)
        for layer, cache in zip(self.layers, ctx_caches, strict=True):
            hidden = layer(
                hidden,
                block_offset,
                cache,
                residual_direction=self._ablation_direction,
                residual_strength=self.config.ablation_strength,
            )
        proposal_hidden = self.norm(hidden)[:, 1:]
        logits = self._project_logits(proposal_hidden)
        unary_proposals = mx.argmax(logits, axis=-1)
        proposals, candidates, sparse_q = self.candidate_selector.select(
            proposal_hidden,
            logits,
            mx.array([pending_token]),
            0.0 if sampling.is_greedy else float(sampling.temperature),
        )
        if sparse_q is None:
            mx.eval(proposals, candidates, unary_proposals)
        else:
            mx.eval(proposals, candidates, sparse_q, unary_proposals)
        self._last_candidate_ids = [
            [int(token) for token in row]
            for row in candidates[0].tolist()
        ]
        self._last_unary_tokens = [
            int(token) for token in unary_proposals[0].tolist()
        ]
        if proposal_policy == "unary":
            proposals = unary_proposals
            sparse_q = None
        proposal_list = [int(token) for token in proposals[0].tolist()]
        distributions = None
        if sparse_q is not None:
            distributions = expand_sparse_candidate_probabilities(
                candidates[0], sparse_q[0], self.config.vocab_size)
            mx.eval(*distributions)
        return proposal_list, distributions, proposal_hidden[0]


class DFlash2SpeculativeDecoder(DSparkSpeculativeDecoder):
    """DFlash proposal source with DSpark's target-authoritative verifier."""

    def __init__(
        self,
        target,
        drafter: DFlash2Drafter,
        *,
        max_draft_tokens: int = MAX_QUANTIZED_PROPOSALS,
        prompt_cache_min_tokens: int = 2048,
        drafter_loader=None,
        release_between_sweeps: bool = True,
        drafter_storage_bytes: int = 0,
        drafter_load_margin_bytes: int = 400_000_000,
        proposal_policy: str = "selector",
        native_mtp_fallback: bool = False,
        fallback_min_dflash_rounds: int = 4,
        fallback_min_accepted_per_round: float = 1.0,
    ):
        if max_draft_tokens > MAX_QUANTIZED_PROPOSALS:
            raise ValueError(
                "quantized MLX DFlash2 is capped at four proposals/block")
        validate_target_compatibility(drafter.config, target.cfg)
        if proposal_policy not in {"selector", "unary"}:
            raise ValueError(
                "DFlash2 proposal_policy must be selector or unary")
        if (isinstance(fallback_min_dflash_rounds, bool)
                or fallback_min_dflash_rounds <= 0):
            raise ValueError(
                "DFlash2 fallback minimum rounds must be positive")
        if (not math.isfinite(fallback_min_accepted_per_round)
                or not 0.0 <= fallback_min_accepted_per_round <= 4.0):
            raise ValueError(
                "DFlash2 fallback acceptance threshold must be in [0, 4]")
        self.proposal_policy = proposal_policy
        self.native_mtp_fallback = bool(native_mtp_fallback)
        self.fallback_min_dflash_rounds = int(fallback_min_dflash_rounds)
        self.fallback_min_accepted_per_round = float(
            fallback_min_accepted_per_round)
        super().__init__(
            target,
            drafter,
            max_draft_tokens=max_draft_tokens,
            confidence_threshold=0.0,
            prompt_cache_min_tokens=prompt_cache_min_tokens,
            context_window_tokens=drafter.config.sliding_window - 1,
            drafter_loader=drafter_loader,
            release_between_sweeps=release_between_sweeps,
            drafter_storage_bytes=drafter_storage_bytes,
            drafter_load_margin_bytes=drafter_load_margin_bytes,
        )
        self.defer_prompt_context_updates = True
        self._candidate_rounds: list[list[list[int]]] = []
        self._unary_rounds: list[list[int]] = []
        self._native_mtp_drafter = None
        if self.native_mtp_fallback:
            from .qwen35_mtp import QwenMTPDrafter

            self._native_mtp_drafter = QwenMTPDrafter(target)
        self._mtp_kv = KVCache(1)
        self._proposal_mode = "dflash2"
        self._round_source = ""
        self._proposal_sources: list[str] = []
        self._dflash_rounds = 0
        self._dflash_proposed = 0
        self._dflash_accepted = 0
        self._native_mtp_rounds = 0
        self._native_mtp_proposed = 0
        self._native_mtp_accepted = 0
        self._fallback_switch_round = 0
        self._native_mtp_load_s = 0.0
        self._native_mtp_release_s = 0.0
        self._native_mtp_read_bytes = 0
        self._native_mtp_loaded_bytes = 0
        self._native_mtp_released_bytes = 0
        self._draft_context_released = False

    def _new_ctx_caches(self) -> list[CtxCache]:
        # Cache containers have no weights. Construct them without making the
        # 1.08 GB draft sidecar resident during target prompt streaming.
        return [CtxCache() for _ in range(self._cfg.num_hidden_layers)]

    def _draft_context_required(self) -> bool:
        return self._proposal_mode == "dflash2"

    @staticmethod
    def _context_to_host(value: mx.array) -> tuple[np.ndarray, str]:
        mx.eval(value)
        if value.dtype == mx.bfloat16:
            return np.array(value.view(mx.uint16), copy=True), "bf16"
        if value.dtype == mx.float16:
            return np.array(value, dtype=np.float16, copy=True), "f16"
        if value.dtype == mx.float32:
            return np.array(value, dtype=np.float32, copy=True), "f32"
        raise TypeError(
            f"unsupported DFlash2 context dtype {value.dtype}")

    @staticmethod
    def _context_from_host(value: np.ndarray, dtype: str) -> mx.array:
        if dtype == "bf16":
            return mx.array(value).view(mx.bfloat16)
        if dtype == "f16":
            return mx.array(value.astype(np.float16, copy=False))
        if dtype == "f32":
            return mx.array(value.astype(np.float32, copy=False))
        raise TypeError(f"unsupported DFlash2 host-context dtype {dtype}")

    def _suspend_draft_context(
        self, ctx_caches: list[CtxCache], stats: DSparkStats,
    ):
        """Losslessly park DFlash-only K/V on the CPU during verification."""
        if not any(cache.k is not None for cache in ctx_caches):
            return None
        if any((cache.k is None) != (cache.v is None) for cache in ctx_caches):
            raise RuntimeError("DFlash2 context has incomplete K/V")
        active_before = int(mx.get_active_memory())
        started = time.perf_counter()
        snapshot = []
        byte_count = 0
        for cache in ctx_caches:
            if cache.k is None:
                snapshot.append((None, None, None, None,
                                 cache.position_start, cache.position_end))
                continue
            host_k, dtype_k = self._context_to_host(cache.k)
            host_v, dtype_v = self._context_to_host(cache.v)
            byte_count += int(host_k.nbytes + host_v.nbytes)
            snapshot.append((host_k, host_v, dtype_k, dtype_v,
                             cache.position_start, cache.position_end))
            cache.k = None
            cache.v = None
        mx.clear_cache()
        stats.draft_context_suspend_rounds += 1
        stats.draft_context_suspend_s += time.perf_counter() - started
        stats.draft_context_suspended_bytes += byte_count
        stats.draft_context_released_active_bytes += max(
            0, active_before - int(mx.get_active_memory()))
        return snapshot

    def _restore_draft_context(
        self, ctx_caches: list[CtxCache], snapshot, stats: DSparkStats,
    ) -> None:
        if snapshot is None:
            return
        if len(snapshot) != len(ctx_caches):
            raise RuntimeError("DFlash2 suspended context layer count changed")
        started = time.perf_counter()
        restored = []
        for cache, entry in zip(ctx_caches, snapshot, strict=True):
            host_k, host_v, dtype_k, dtype_v, position_start, position_end = entry
            if cache.k is not None or cache.v is not None:
                raise RuntimeError(
                    "DFlash2 context became resident while suspended")
            cache.position_start = position_start
            cache.position_end = int(position_end)
            if host_k is not None:
                cache.k = self._context_from_host(host_k, dtype_k)
                cache.v = self._context_from_host(host_v, dtype_v)
                restored.extend((cache.k, cache.v))
        if restored:
            mx.eval(*restored)
        stats.draft_context_restore_rounds += 1
        stats.draft_context_restore_s += time.perf_counter() - started

    def _discard_suspended_draft_context(
        self, ctx_caches: list[CtxCache], snapshot,
    ) -> None:
        del snapshot
        self._release_context_caches(ctx_caches)
        self._draft_context_released = True

    def _note_verified_round(self, proposed: int, accepted: int) -> None:
        if proposed <= 0:
            return
        if self._round_source == "D":
            self._dflash_rounds += 1
            self._dflash_proposed += int(proposed)
            self._dflash_accepted += int(accepted)
            if (
                self.native_mtp_fallback
                and self._dflash_rounds >= self.fallback_min_dflash_rounds
                and self._dflash_accepted < (
                    self.fallback_min_accepted_per_round
                    * self._dflash_rounds)
            ):
                self._proposal_mode = "native-mtp"
                self._fallback_switch_round = len(self._proposal_sources)
        elif self._round_source == "M":
            self._native_mtp_rounds += 1
            self._native_mtp_proposed += int(proposed)
            self._native_mtp_accepted += int(accepted)

    @staticmethod
    def _release_context_caches(ctx_caches: list[CtxCache]) -> None:
        for cache in ctx_caches:
            cache.k = None
            cache.v = None
            cache.position_start = None
            cache.position_end = 0
        mx.clear_cache()

    def _propose_native_mtp(
        self,
        pending: int,
        offset: int,
        ctx_caches: list[CtxCache],
        sampling: SamplingParams,
        history: list[int],
    ):
        if self._native_mtp_drafter is None:
            raise RuntimeError("DFlash2 native-MTP fallback was not initialized")
        if not self._draft_context_released:
            self._release_context_caches(ctx_caches)
            self._draft_context_released = True

        drafter = self._native_mtp_drafter
        prepare = getattr(drafter, "prepare_request_weights", None)
        release = getattr(drafter, "release_request_weights", None)
        cache_stats = getattr(self.target.cache, "stats", None)
        read_before = int(getattr(cache_stats, "bytes_read", 0))
        weights = None
        load_started = time.perf_counter()
        try:
            weights = prepare() if callable(prepare) else None
            self._native_mtp_load_s += time.perf_counter() - load_started
            if weights is not None:
                self._native_mtp_loaded_bytes += sum(
                    int(getattr(value, "nbytes", 0))
                    for value in weights.values()
                )
            logits, _hidden = drafter.draft_step(
                self.target._h_last,
                pending,
                self._mtp_kv,
                offset - 1,
                weights,
            )
            if sampling.is_greedy:
                token = int(mx.argmax(logits).item())
                distributions = None
            else:
                from .qwen35_mtp import _flat_top_k_draft_probabilities

                q = _flat_top_k_draft_probabilities(
                    logits, sampling, history, 1)
                token = int(sample_probabilities(q))
                distributions = [q]
        finally:
            if weights is not None:
                release_started = time.perf_counter()
                if not callable(release):
                    weights.clear()
                    mx.clear_cache()
                    raise RuntimeError(
                        "native-MTP fallback omits round weight release")
                release_info = release(weights) or {}
                self._native_mtp_release_s += (
                    time.perf_counter() - release_started)
                self._native_mtp_released_bytes += int(
                    release_info.get("resident_bytes", 0))
        self._native_mtp_read_bytes += max(
            0, int(getattr(cache_stats, "bytes_read", 0)) - read_before)
        self._round_source = "M"
        self._proposal_sources.append("M")
        self._candidate_rounds.append([[token]])
        self._unary_rounds.append([token])
        return [token], distributions

    def _propose(
        self,
        pending: int,
        offset: int,
        ctx_caches: list[CtxCache],
        cap: int,
        sampling: SamplingParams,
        history: list[int],
    ):
        if self._proposal_mode == "native-mtp":
            return self._propose_native_mtp(
                pending, offset, ctx_caches, sampling, history)
        del history  # DFlash2 q is selector-conditional, not repetition-filtered.
        proposals, distributions, _hidden = self._ensure_drafter().propose_block(
            pending, offset, ctx_caches, cap, sampling,
            proposal_policy=self.proposal_policy)
        self._candidate_rounds.append([
            list(row)
            for row in self._ensure_drafter()._last_candidate_ids
        ])
        self._unary_rounds.append(list(
            self._ensure_drafter()._last_unary_tokens))
        self._round_source = "D"
        self._proposal_sources.append("D")
        return proposals, distributions

    def generate(self, *args, **kwargs):
        self._candidate_rounds.clear()
        self._unary_rounds.clear()
        self._mtp_kv = KVCache(1)
        self._proposal_mode = "dflash2"
        self._round_source = ""
        self._proposal_sources.clear()
        self._dflash_rounds = 0
        self._dflash_proposed = 0
        self._dflash_accepted = 0
        self._native_mtp_rounds = 0
        self._native_mtp_proposed = 0
        self._native_mtp_accepted = 0
        self._fallback_switch_round = 0
        self._native_mtp_load_s = 0.0
        self._native_mtp_release_s = 0.0
        self._native_mtp_read_bytes = 0
        self._native_mtp_loaded_bytes = 0
        self._native_mtp_released_bytes = 0
        self._draft_context_released = False
        result = super().generate(*args, **kwargs)
        # For greedy decoding, the committed stream plus each round's accepted
        # prefix reconstructs every target decision that influenced progress.
        # This lets us distinguish a selector miss from a base top-k miss
        # without retaining logits or target hidden states.
        candidate_positions = candidate_hits = 0
        unary_positions = unary_hits = 0
        emitted = [int(token) for token in result.get("tokens", ())]
        stats = result.get("stats")
        rounds = list(getattr(stats, "rounds", ()))
        if kwargs.get("sampling") is None or kwargs["sampling"].is_greedy:
            candidate_positions, candidate_hits = greedy_candidate_recall(
                emitted, rounds, self._candidate_rounds)
            unary_positions, unary_hits = greedy_candidate_recall(
                emitted, rounds,
                [[[token] for token in row] for row in self._unary_rounds])
        result.setdefault("path_stats", {}).update({
            "speculative_kind": "dflash2",
            "dflash2_enabled": 1,
            "dflash2_checkpoint_block_size": self._cfg.block_size,
            "dflash2_effective_block_size": self.max_draft_tokens + 1,
            "dflash2_selector_top_k": self._cfg.selector_top_k,
            "dflash2_context_window_tokens": self.context_window_tokens,
            "dflash2_target_verifier": "dspark-qwen-authoritative",
            "dflash2_proposal_policy": self.proposal_policy,
            "dflash2_fused_dynamic_conv": int(
                self._cfg.fused_dynamic_conv),
            "dflash2_ablation_enabled": int(self._cfg.ablation_enabled),
            "dflash2_ablation_strength": self._cfg.ablation_strength,
            "dflash2_ablation_fingerprint": self._cfg.ablation_fingerprint,
            "dflash2_candidate_recall_positions": candidate_positions,
            "dflash2_candidate_recall_hits": candidate_hits,
            "dflash2_candidate_recall": (
                candidate_hits / candidate_positions
                if candidate_positions else None),
            "dflash2_unary_recall_positions": unary_positions,
            "dflash2_unary_recall_hits": unary_hits,
            "dflash2_unary_recall": (
                unary_hits / unary_positions if unary_positions else None),
            "dflash2_native_mtp_fallback_enabled": int(
                self.native_mtp_fallback),
            "dflash2_native_mtp_fallback_switched": int(
                self._fallback_switch_round > 0),
            "dflash2_native_mtp_fallback_switch_round": (
                self._fallback_switch_round),
            "dflash2_native_mtp_fallback_min_rounds": (
                self.fallback_min_dflash_rounds),
            "dflash2_native_mtp_fallback_min_accepted_per_round": (
                self.fallback_min_accepted_per_round),
            "dflash2_proposal_sources": "".join(self._proposal_sources),
            "dflash2_rounds": self._dflash_rounds,
            "dflash2_proposed": self._dflash_proposed,
            "dflash2_accepted": self._dflash_accepted,
            "dflash2_fallback_native_mtp_rounds": self._native_mtp_rounds,
            "dflash2_fallback_native_mtp_proposed": (
                self._native_mtp_proposed),
            "dflash2_fallback_native_mtp_accepted": (
                self._native_mtp_accepted),
            "dflash2_fallback_native_mtp_load_s": self._native_mtp_load_s,
            "dflash2_fallback_native_mtp_release_s": (
                self._native_mtp_release_s),
            "dflash2_fallback_native_mtp_read_bytes": (
                self._native_mtp_read_bytes),
            "dflash2_fallback_native_mtp_loaded_bytes": (
                self._native_mtp_loaded_bytes),
            "dflash2_fallback_native_mtp_released_bytes": (
                self._native_mtp_released_bytes),
        })
        return result


class DFlash2SpeculativeEngine:
    """Explicit engine wrapper; intentionally absent from server dispatch."""

    def __init__(
        self,
        target,
        draft_dir: str | Path,
        *,
        max_draft_tokens: int = MAX_QUANTIZED_PROPOSALS,
        max_prompt_tokens: int = 262_144,
        prompt_cache_min_tokens: int = 2048,
        release_between_sweeps: bool = True,
        drafter_load_margin_bytes: int = 400_000_000,
        verify_sidecar_hash: bool = True,
        proposal_policy: str = "selector",
        fused_dynamic_conv: bool = False,
        ablation_direction_dir: str | Path | None = None,
        ablation_strength: float = 1.0,
        native_mtp_fallback: bool = False,
        fallback_min_dflash_rounds: int = 4,
        fallback_min_accepted_per_round: float = 1.0,
    ):
        if max_prompt_tokens <= 0:
            raise ValueError("DFlash2 max_prompt_tokens must be positive")
        self.target = target
        self.max_prompt_tokens = max_prompt_tokens
        self._closed = False
        draft_dir = Path(draft_dir)
        draft_bytes = (draft_dir / "model.safetensors").stat().st_size
        ablation = None
        if ablation_direction_dir:
            target_dir = Path(getattr(target, "_model_dir", ""))
            target_config = target_dir / "config.json"
            if not target_config.is_file():
                raise ValueError(
                    "DFlash2 ablation requires the target checkpoint config")
            ablation = load_ablation_artifact(
                ablation_direction_dir,
                target_config=target_config,
                draft_revision=OFFICIAL_REVISION,
                hidden_size=target.cfg.hidden_size,
            )

        def bind(drafter):
            drafter.bind_target_lm_head(target._lm_head_weight)
            drafter.bind_target_embed(
                lambda token_ids: target._embed(list(token_ids)))
            if ablation is not None:
                drafter.set_ablation(
                    mx.array(ablation.direction),
                    strength=ablation_strength,
                    fingerprint=ablation.fingerprint,
                )
            return drafter

        def load_drafter():
            prepare_for = getattr(target.cache, "prepare_for", None)
            if callable(prepare_for):
                prepare_for(draft_bytes)
            if target.governor is not None:
                target.governor.reserve(
                    draft_bytes,
                    margin=drafter_load_margin_bytes,
                    reason="dflash2-sidecar-load",
                )
            # The first construction below verified the content hash.  Round
            # reloads validate metadata/header identity without rereading the
            # entire 1.08 GB artifact solely to hash it again.
            return bind(DFlash2Drafter.load(
                draft_dir,
                verify_hash=False,
                fused_dynamic_conv=fused_dynamic_conv,
            ))

        if target.governor is not None:
            target.governor.reserve(
                draft_bytes, reason="dflash2-sidecar-initial-load")
        drafter = bind(DFlash2Drafter.load(
            draft_dir,
            verify_hash=verify_sidecar_hash,
            fused_dynamic_conv=fused_dynamic_conv,
        ))
        if target.governor is not None:
            target.governor.fit_cache_to_live_headroom()
        self.decoder = DFlash2SpeculativeDecoder(
            target,
            drafter,
            max_draft_tokens=max_draft_tokens,
            prompt_cache_min_tokens=prompt_cache_min_tokens,
            drafter_loader=load_drafter,
            release_between_sweeps=release_between_sweeps,
            drafter_storage_bytes=draft_bytes,
            drafter_load_margin_bytes=drafter_load_margin_bytes,
            proposal_policy=proposal_policy,
            native_mtp_fallback=native_mtp_fallback,
            fallback_min_dflash_rounds=fallback_min_dflash_rounds,
            fallback_min_accepted_per_round=(
                fallback_min_accepted_per_round),
        )
        self._speculative_k = self.decoder.max_draft_tokens
        self._speculative_draft_dir = draft_dir
        self._speculative_kind = "dflash2"
        # Construction validates and binds the artifact. Every request starts
        # target prefill without the sidecar, even when decode-resident mode is
        # explicitly selected to retain it between verification rounds.
        self.decoder._release_drafter(force=True)

    def __getattr__(self, name):
        return getattr(self.target, name)

    def _target_generate(self, reason: str, prompt: str, max_tokens: int,
                         **kwargs):
        self.decoder.clear_prompt_cache()
        self.decoder._release_drafter(force=True)
        generate = getattr(
            self.target, "generate_with_memory_retry", self.target.generate)
        result = generate(prompt, max_tokens, **kwargs)
        result.setdefault("path_stats", {}).update({
            "speculative_enabled": 1,
            "speculative_used": 0,
            "speculative_kind": "dflash2",
            "speculative_fallback_reason": reason,
            "speculative_k": self._speculative_k,
        })
        return result

    def generate(self, prompt: str, max_tokens: int = 64, on_token=None,
                 stop=None, on_progress=None,
                 sampling: SamplingParams | None = None,
                 constraint=None):
        prepared = getattr(prompt, "token_ids", None)
        ids = (list(prepared) if prepared is not None
               else list(self.target.tokenizer.encode(prompt).ids))
        kwargs = {
            "on_token": on_token,
            "stop": stop,
            "on_progress": on_progress,
            "sampling": sampling,
            "constraint": constraint,
        }
        kwargs = {name: value for name, value in kwargs.items()
                  if value is not None}
        # A previous request may have retained the proposal weights for decode.
        # Target prefill always gets an isolated lifetime.
        self.decoder._release_drafter(force=True)
        if len(ids) > self.max_prompt_tokens:
            return self._target_generate(
                "prompt-limit", prompt, max_tokens, **kwargs)
        try:
            return self.decoder.generate(
                prompt,
                max_tokens,
                encoded_ids=ids,
                **kwargs,
            )
        finally:
            self.decoder._release_drafter()

    def release_request_state(self):
        self.decoder.clear_prompt_cache()
        self.decoder._release_drafter(force=True)
        self.target.release_request_state()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.decoder._release_drafter(force=True)
        close = getattr(self.target, "close", None)
        if callable(close):
            close()
