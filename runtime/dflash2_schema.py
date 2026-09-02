"""Pure metadata contracts for pinned Qwen3.8 and GLM-5.3-Flash drafts.

This module deliberately imports neither MLX nor the serving runtime.  Phase A
only establishes what the published checkpoint *is* and rejects a partial or
silently downgraded DFlash/DFlash2 artifact before any tensor is materialized.

The pinned values below were obtained from Hugging Face repository metadata
and safetensors header-only range reads. Each source identity is independently
hash-pinned; similar DFlash2 geometry never substitutes for release identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


OFFICIAL_REPOSITORY = "incoai/Qwen3.8-27B-DFlash2"
OFFICIAL_REVISION = "dedf8df68adfb1afeaf7b7480c0a0243108177b4"
OFFICIAL_CONFIG_SHA256 = (
    "873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980")
OFFICIAL_WEIGHTS_SHA256 = (
    "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c")
OFFICIAL_WEIGHTS_BYTES = 3_848_817_896
OFFICIAL_PARAMETER_COUNT = 1_924_404_480
OFFICIAL_UPSTREAM_REPOSITORY = "https://github.com/z-lab/dflash"
OFFICIAL_UPSTREAM_REVISION = "07ebd93db9f472af339b644bb70221ad8428328a"

GLM53_FLASH_REPOSITORY = "incoai/GLM-5.3-Flash-DFlash2"
GLM53_FLASH_REVISION = "bf582e4eacc1810f76656d1811693ff6c6737d2a"
GLM53_FLASH_CONFIG_SHA256 = (
    "c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573")
GLM53_FLASH_WEIGHTS_SHA256 = (
    "b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac")
GLM53_FLASH_WEIGHTS_BYTES = 2_342_169_800
GLM53_FLASH_PARAMETER_COUNT = 1_171_080_448


# Kept as a literal receipt so ``plan`` remains useful before the large shard
# exists locally.  The config file's byte hash above is still authoritative.
OFFICIAL_CONFIG: dict[str, Any] = {
    "architectures": ["DFlash2DraftModel"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": None,
    "is_causal": False,
    "dflash_config": {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 248070,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [5, 19, 33, 47, 61],
    },
    "dtype": "bfloat16",
    "eos_token_id": 248044,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "layer_types": ["sliding_attention"] * 5,
    "max_position_embeddings": 262144,
    "max_window_layers": 5,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 5,
    "num_key_value_heads": 8,
    "num_target_layers": 64,
    "pad_token_id": 248044,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {
        "rope_theta": 10_000_000,
        "rope_type": "default",
    },
    "sliding_window": 2048,
    "tie_word_embeddings": False,
    "transformers_version": "5.15.0",
    "use_cache": True,
    "use_sliding_window": True,
    "vocab_size": 248320,
}

GLM53_FLASH_CONFIG: dict[str, Any] = {
    "architectures": ["DFlash2DraftModel"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": None,
    "dflash_config": {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 154856,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [5, 14, 24, 33, 42],
    },
    "dtype": "bfloat16",
    "eos_token_id": [154820, 154827, 154829],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "is_causal": False,
    "layer_types": ["sliding_attention"] * 5,
    "max_position_embeddings": 1048576,
    "max_window_layers": 5,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 5,
    "num_key_value_heads": 8,
    "num_target_layers": 45,
    "pad_token_id": 154820,
    "rms_norm_eps": 1e-5,
    "rope_parameters": {
        "rope_theta": 10000.0,
        "rope_type": "default",
    },
    "sliding_window": 2048,
    "tie_word_embeddings": False,
    "transformers_version": "5.7.0",
    "use_cache": False,
    "use_sliding_window": True,
    "vocab_size": 154880,
}


@dataclass(frozen=True)
class DFlash2Release:
    variant: str
    repository: str
    revision: str
    config_sha256: str
    weights_sha256: str
    weights_bytes: int
    parameter_count: int
    config: Mapping[str, Any]
    target_model_type: str


QWEN38_RELEASE = DFlash2Release(
    variant="qwen38",
    repository=OFFICIAL_REPOSITORY,
    revision=OFFICIAL_REVISION,
    config_sha256=OFFICIAL_CONFIG_SHA256,
    weights_sha256=OFFICIAL_WEIGHTS_SHA256,
    weights_bytes=OFFICIAL_WEIGHTS_BYTES,
    parameter_count=OFFICIAL_PARAMETER_COUNT,
    config=OFFICIAL_CONFIG,
    target_model_type="qwen3_5",
)
GLM53_FLASH_RELEASE = DFlash2Release(
    variant="glm53-flash",
    repository=GLM53_FLASH_REPOSITORY,
    revision=GLM53_FLASH_REVISION,
    config_sha256=GLM53_FLASH_CONFIG_SHA256,
    weights_sha256=GLM53_FLASH_WEIGHTS_SHA256,
    weights_bytes=GLM53_FLASH_WEIGHTS_BYTES,
    parameter_count=GLM53_FLASH_PARAMETER_COUNT,
    config=GLM53_FLASH_CONFIG,
    target_model_type="glm5_next",
)


def release_for_variant(variant: str) -> DFlash2Release:
    releases = {
        QWEN38_RELEASE.variant: QWEN38_RELEASE,
        GLM53_FLASH_RELEASE.variant: GLM53_FLASH_RELEASE,
    }
    try:
        return releases[str(variant)]
    except KeyError as error:
        raise ValueError(f"unsupported DFlash2 release variant {variant!r}") from error


def release_for_repository(repository: str) -> DFlash2Release:
    for release in (QWEN38_RELEASE, GLM53_FLASH_RELEASE):
        if repository == release.repository:
            return release
    raise ValueError(f"unsupported official DFlash2 repository {repository!r}")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def require_revision(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("source revision must be a 40-character lowercase SHA")
    return value


def _integer(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        if self.dtype != "BF16":
            raise ValueError(f"unsupported DFlash2 source dtype {self.dtype!r}")
        return self.elements * 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "bytes": self.nbytes,
        }


@dataclass(frozen=True)
class DFlash2Config:
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    num_target_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    max_position_embeddings: int
    sliding_window: int
    block_size: int
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    layer_types: tuple[str, ...]
    rope_theta: float
    rope_type: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DFlash2Config":
        if not isinstance(raw, Mapping):
            raise ValueError("DFlash2 config must be a JSON object")
        if raw.get("architectures") != ["DFlash2DraftModel"]:
            raise ValueError(
                "DFlash2 architecture must be exactly ['DFlash2DraftModel']; "
                "refusing a DFlash1 or ambiguous checkpoint")
        if raw.get("model_type") != "qwen3":
            raise ValueError("Qwen DFlash2 model_type must be 'qwen3'")
        if raw.get("is_causal") is not False:
            raise ValueError(
                "Qwen DFlash2 is_causal must be explicit false; a missing/true "
                "value silently changes the published draft architecture")
        if raw.get("dtype") != "bfloat16":
            raise ValueError("released DFlash2 source dtype must be bfloat16")
        if raw.get("attention_bias") is not False:
            raise ValueError("DFlash2 attention_bias must be false")
        if raw.get("hidden_act") != "silu":
            raise ValueError("DFlash2 hidden_act must be 'silu'")
        if raw.get("tie_word_embeddings") is not False:
            raise ValueError("DFlash2 must reuse an external untied target head")

        dflash = raw.get("dflash_config")
        if not isinstance(dflash, Mapping):
            raise ValueError("DFlash2 config has no dflash_config object")
        rope = raw.get("rope_parameters")
        if not isinstance(rope, Mapping):
            raise ValueError("DFlash2 config has no rope_parameters object")

        hidden_size = _integer(raw.get("hidden_size"), "hidden_size")
        heads = _integer(raw.get("num_attention_heads"), "num_attention_heads")
        kv_heads = _integer(
            raw.get("num_key_value_heads"), "num_key_value_heads")
        head_dim = _integer(raw.get("head_dim"), "head_dim")
        layers = _integer(raw.get("num_hidden_layers"), "num_hidden_layers")
        target_layers = _integer(
            raw.get("num_target_layers"), "num_target_layers")
        group_size = _integer(
            dflash.get("conv_group_size"), "dflash_config.conv_group_size")
        # Qwen3.8's draft attention width is 32*128=4096 while the residual
        # stream is 5120; o_proj bridges those widths.  Do not import the
        # common but false hidden==heads*head_dim Transformer assumption here.
        if heads % kv_heads:
            raise ValueError("num_attention_heads must be divisible by KV heads")
        if hidden_size % group_size:
            raise ValueError("hidden_size must be divisible by conv_group_size")

        layer_types_raw = raw.get("layer_types")
        if not isinstance(layer_types_raw, list) or len(layer_types_raw) != layers:
            raise ValueError("layer_types must have one entry per draft layer")
        layer_types = tuple(layer_types_raw)
        if set(layer_types) != {"sliding_attention"}:
            raise ValueError(
                "this Qwen DFlash2 contract requires every draft layer to use "
                "sliding_attention")
        taps_raw = dflash.get("target_layer_ids")
        if not isinstance(taps_raw, list) or not taps_raw:
            raise ValueError("DFlash2 target_layer_ids must be a non-empty list")
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in taps_raw):
            raise ValueError("DFlash2 target_layer_ids must contain integers")
        taps = tuple(taps_raw)
        if tuple(sorted(set(taps))) != taps:
            raise ValueError("DFlash2 target_layer_ids must be unique and increasing")
        if taps[0] < 0 or taps[-1] >= target_layers:
            raise ValueError("DFlash2 target_layer_ids exceed the target depth")

        vocab = _integer(raw.get("vocab_size"), "vocab_size")
        mask = _integer(dflash.get("mask_token_id"), "mask_token_id", minimum=0)
        if mask >= vocab:
            raise ValueError("DFlash2 mask_token_id must be inside the vocabulary")
        selector_top_k = _integer(
            dflash.get("selector_top_k"), "dflash_config.selector_top_k")
        if selector_top_k > vocab:
            raise ValueError("DFlash2 selector_top_k exceeds vocabulary size")

        config = cls(
            hidden_size=hidden_size,
            intermediate_size=_integer(
                raw.get("intermediate_size"), "intermediate_size"),
            vocab_size=vocab,
            num_hidden_layers=layers,
            num_target_layers=target_layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            rms_norm_eps=_number(raw.get("rms_norm_eps"), "rms_norm_eps"),
            max_position_embeddings=_integer(
                raw.get("max_position_embeddings"),
                "max_position_embeddings"),
            sliding_window=_integer(raw.get("sliding_window"), "sliding_window"),
            block_size=_integer(
                dflash.get("block_size"), "dflash_config.block_size", minimum=2),
            conv_kernel_size=_integer(
                dflash.get("conv_kernel_size"),
                "dflash_config.conv_kernel_size"),
            conv_group_size=group_size,
            selector_rank=_integer(
                dflash.get("selector_rank"), "dflash_config.selector_rank"),
            selector_top_k=selector_top_k,
            mask_token_id=mask,
            target_layer_ids=taps,
            layer_types=layer_types,
            rope_theta=_number(rope.get("rope_theta"), "rope_theta"),
            rope_type=str(rope.get("rope_type", "")),
        )
        if config.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if config.rope_theta <= 0 or config.rope_type != "default":
            raise ValueError("DFlash2 requires positive default RoPE")
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "DFlash2Config":
        raw = json.loads(Path(path).read_text())
        return cls.from_mapping(raw)

    def validate_official_qwen38(self) -> None:
        expected = DFlash2Config.from_mapping(OFFICIAL_CONFIG)
        if self != expected:
            fields = [
                name for name in self.__dataclass_fields__
                if getattr(self, name) != getattr(expected, name)]
            raise ValueError(
                "DFlash2 config does not match pinned Qwen3.8 geometry: "
                + ", ".join(fields))

    def validate_official_glm53_flash(self) -> None:
        expected = DFlash2Config.from_mapping(GLM53_FLASH_CONFIG)
        if self != expected:
            fields = [
                name for name in self.__dataclass_fields__
                if getattr(self, name) != getattr(expected, name)]
            raise ValueError(
                "DFlash2 config does not match pinned GLM-5.3-Flash "
                "geometry: " + ", ".join(fields))

    def validate_official_release(self, release: DFlash2Release) -> None:
        expected = DFlash2Config.from_mapping(release.config)
        if self != expected:
            fields = [
                name for name in self.__dataclass_fields__
                if getattr(self, name) != getattr(expected, name)]
            raise ValueError(
                f"DFlash2 config does not match pinned {release.variant} "
                "geometry: " + ", ".join(fields))

    @property
    def proposal_count(self) -> int:
        return self.block_size - 1

    def expected_tensor_specs(self) -> dict[str, TensorSpec]:
        hidden = self.hidden_size
        heads_width = self.num_attention_heads * self.head_dim
        kv_width = self.num_key_value_heads * self.head_dim
        groups = hidden // self.conv_group_size
        kernel_projection = 2 * self.conv_kernel_size * groups
        specs = {
            "candidate_selector.hidden_projection.weight": TensorSpec(
                "BF16", (self.selector_rank, hidden)),
            "candidate_selector.predecessor_codebook": TensorSpec(
                "BF16", (self.vocab_size, self.selector_rank)),
            "candidate_selector.successor_codebook": TensorSpec(
                "BF16", (self.vocab_size, self.selector_rank)),
            "fc.weight": TensorSpec(
                "BF16", (hidden, len(self.target_layer_ids) * hidden)),
            "hidden_norm.weight": TensorSpec("BF16", (hidden,)),
            "norm.weight": TensorSpec("BF16", (hidden,)),
        }
        for layer in range(self.num_hidden_layers):
            prefix = f"layers.{layer}"
            specs.update({
                f"{prefix}.attention_conv.base_kernel": TensorSpec(
                    "BF16", (2, self.conv_kernel_size, hidden)),
                f"{prefix}.attention_conv.kernel_projection.weight": TensorSpec(
                    "BF16", (kernel_projection, hidden)),
                f"{prefix}.input_layernorm.weight": TensorSpec(
                    "BF16", (hidden,)),
                f"{prefix}.mlp.down_proj.weight": TensorSpec(
                    "BF16", (hidden, self.intermediate_size)),
                f"{prefix}.mlp.gate_proj.weight": TensorSpec(
                    "BF16", (self.intermediate_size, hidden)),
                f"{prefix}.mlp.up_proj.weight": TensorSpec(
                    "BF16", (self.intermediate_size, hidden)),
                f"{prefix}.mlp_conv.base_kernel": TensorSpec(
                    "BF16", (2, self.conv_kernel_size, hidden)),
                f"{prefix}.mlp_conv.kernel_projection.weight": TensorSpec(
                    "BF16", (kernel_projection, hidden)),
                f"{prefix}.post_attention_layernorm.weight": TensorSpec(
                    "BF16", (hidden,)),
                f"{prefix}.self_attn.k_norm.weight": TensorSpec(
                    "BF16", (self.head_dim,)),
                f"{prefix}.self_attn.k_proj.weight": TensorSpec(
                    "BF16", (kv_width, hidden)),
                f"{prefix}.self_attn.o_proj.weight": TensorSpec(
                    "BF16", (hidden, heads_width)),
                f"{prefix}.self_attn.q_norm.weight": TensorSpec(
                    "BF16", (self.head_dim,)),
                f"{prefix}.self_attn.q_proj.weight": TensorSpec(
                    "BF16", (heads_width, hidden)),
                f"{prefix}.self_attn.v_proj.weight": TensorSpec(
                    "BF16", (kv_width, hidden)),
            })
        return dict(sorted(specs.items()))

    @property
    def source_parameter_count(self) -> int:
        return sum(spec.elements for spec in self.expected_tensor_specs().values())


def _read_safetensors_header(path: str | Path) -> tuple[dict[str, Any], int, int]:
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 0 or header_length > size - 8:
            raise ValueError(
                f"invalid safetensors header length {header_length}: {path}")
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"truncated safetensors header JSON: {path}")
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid safetensors header JSON: {path}") from error
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must be an object: {path}")
    return header, header_length, size


def validate_source_header(
    config: DFlash2Config,
    header: Mapping[str, Any],
    *,
    payload_bytes: int | None = None,
) -> dict[str, Any]:
    """Validate the complete tensor allowlist without loading any tensor."""
    if not isinstance(header, Mapping):
        raise ValueError("safetensors header must be an object")
    metadata = header.get("__metadata__")
    if metadata not in (None, {}):
        if not isinstance(metadata, Mapping):
            raise ValueError("invalid safetensors __metadata__")
    actual_names = {name for name in header if name != "__metadata__"}
    expected = config.expected_tensor_specs()
    expected_names = set(expected)
    if actual_names != expected_names:
        raise ValueError(
            "DFlash2 source tensor set mismatch: "
            f"missing={sorted(expected_names - actual_names)[:8]} "
            f"unexpected={sorted(actual_names - expected_names)[:8]}")

    offsets: list[tuple[int, int, str]] = []
    normalized: dict[str, dict[str, Any]] = {}
    for name, expected_spec in expected.items():
        raw = header[name]
        if not isinstance(raw, Mapping):
            raise ValueError(f"invalid safetensors entry for {name}")
        shape = raw.get("shape")
        dtype = raw.get("dtype")
        span = raw.get("data_offsets")
        if not (
            isinstance(shape, list)
            and all(isinstance(value, int) and value >= 0 for value in shape)
            and isinstance(dtype, str)
            and isinstance(span, list) and len(span) == 2
            and all(isinstance(value, int) for value in span)
            and 0 <= span[0] <= span[1]
        ):
            raise ValueError(f"invalid safetensors metadata for {name}")
        actual = TensorSpec(dtype, tuple(shape))
        if actual != expected_spec:
            raise ValueError(
                f"DFlash2 tensor {name} mismatch: "
                f"expected {expected_spec.as_dict()}, got {actual.as_dict()}")
        if span[1] - span[0] != expected_spec.nbytes:
            raise ValueError(f"DFlash2 tensor {name} byte span is inconsistent")
        offsets.append((span[0], span[1], name))
        normalized[name] = expected_spec.as_dict()

    cursor = 0
    for start, end, name in sorted(offsets):
        if start != cursor:
            raise ValueError(
                f"DFlash2 safetensors payload is not contiguous before {name}: "
                f"expected offset {cursor}, got {start}")
        cursor = end
    if payload_bytes is not None and cursor != payload_bytes:
        raise ValueError(
            f"DFlash2 payload byte mismatch: tensors cover {cursor}, "
            f"file contains {payload_bytes}")

    return {
        "tensor_count": len(expected),
        "parameter_count": config.source_parameter_count,
        "tensor_bytes": cursor,
        "tensor_schema_sha256": sha256_bytes(canonical_json(normalized)),
    }


def inspect_source_file(config: DFlash2Config, path: str | Path) -> dict[str, Any]:
    header, header_length, file_bytes = _read_safetensors_header(path)
    report = validate_source_header(
        config, header, payload_bytes=file_bytes - 8 - header_length)
    return {
        **report,
        "file_bytes": file_bytes,
        "header_bytes": header_length,
        "payload_bytes": file_bytes - 8 - header_length,
    }
