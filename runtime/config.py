"""Model configuration loaded from a HF-style config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _validated_token_ids(value, vocab_size: int, label: str) -> tuple[int, ...]:
    """Normalize one HF token-id scalar/list without coercing invalid JSON."""
    if value is None:
        return ()
    values = [value] if isinstance(value, int) and not isinstance(value, bool) else value
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be an integer or an array of integers")
    normalized = []
    for token_id in values:
        if (not isinstance(token_id, int) or isinstance(token_id, bool)
                or not 0 <= token_id < vocab_size):
            raise ValueError(
                f"{label} contains invalid token id {token_id!r}; "
                f"expected an integer in [0, {vocab_size})")
        if token_id not in normalized:
            normalized.append(token_id)
    return tuple(normalized)


def _validated_pixel_bound(value, default: int, label: str) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass
class ModelConfig:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    head_dim: int
    eos_token_ids: tuple[int, ...]
    torch_dtype: str
    num_experts: int = 0  # 0 = dense model
    num_experts_per_tok: int = 0
    # Optional runtime-only lossy experiment: per-layer active expert counts.
    # Checkpoints leave this empty and retain their released global top-k.
    expert_top_k_by_layer: tuple[int, ...] = ()
    moe_intermediate_size: int = 0  # per-expert MLP width when it differs from dense (GLM: 2048 vs 12288)
    norm_topk_prob: bool = False
    # gpt-oss specifics
    layer_types: tuple = ()  # per-layer "sliding_attention" | "full_attention"
    sliding_window: int = 0
    swiglu_limit: float = 7.0
    rope_scaling: dict | None = None  # yarn params
    # GLM-5.2 / DeepSeek-style MLA + MoE specifics
    first_k_dense_replace: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    n_shared_experts: int = 0
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    mlp_layer_types: tuple = ()
    num_nextn_predict_layers: int = 0
    rope_interleave: bool = False
    # GLM's low-rank Q/KV RMSNorm modules are constructed with the architecture
    # default eps=1e-6, independently of the 1e-5 decoder-block norm epsilon.
    # Keeping this explicit prevents a seemingly harmless config-wide epsilon
    # substitution from changing released BF16 activations.
    mla_latent_norm_eps: float = 1e-6
    index_topk: int = 0  # DSA: top-k attended positions (dense-exact below this)
    index_head_dim: int = 128
    index_n_heads: int = 32
    indexer_types: tuple = ()  # per-layer 'full' | 'shared' (IndexShare)
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 2
    index_share_for_mtp_iteration: bool = False
    # Qwen3-VL vision extras (None/0 for text-only models)
    vision_config: dict | None = None
    image_token_id: int = 0
    video_token_id: int = 0
    vision_start_token_id: int = 0
    vision_end_token_id: int = 0
    # Qwen3-VL's video processor applies these bounds to T*H*W, not to each
    # frame independently. Defaults are the released architecture values;
    # video_preprocessor_config.json overrides them when present.
    video_min_pixels: int = 4_096
    video_max_pixels: int = 25_165_824
    # Kimi Linear (KDA) hybrid attention layout — see docs/future_lossless_techniques.md F92.
    # Both tuples are 0-indexed layer numbers; a layer must appear in exactly one.
    kda_layers: tuple[int, ...] = ()
    full_attn_layers: tuple[int, ...] = ()
    kda_head_dim: int = 0
    kda_num_heads: int = 0
    kda_conv_kernel_size: int = 4
    moe_layer_freq: int = 1
    # F92/F93: expert tensor prefix under each layer. GLM/gpt-oss/OLMoE/
    # generic Mixtral-style checkpoints use "mlp.experts"; Kimi's MoE module
    # is named "block_sparse_moe" instead. Every expert-fetch call site in
    # engine.py must use this, not a hardcoded ".mlp.experts." substring, or
    # MoE paging silently finds zero matching tensors for Kimi models.
    moe_expert_prefix: str = "mlp.experts"
    # F92: Kimi Linear's MLA layers are NoPE -- config.json's mla_use_nope=true
    # means RoPE is never applied to the "rope" head-dim split at all (real
    # modeling_kimi.py has no apply_rotary_emb call anywhere; position info
    # comes only from the KDA layers' inherent sequential recurrence). GLM-5.2
    # always applies real RoPE; default False preserves that unchanged.
    mla_use_nope: bool = False

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "ModelConfig":
        # F24: NAS-hosted models — retry config reads through transient SMB drops
        # (engine init died to exactly this once).
        model_dir = Path(model_dir)
        path = model_dir / "config.json"
        for attempt in range(4):
            try:
                raw = json.loads(path.read_text())
                break
            except OSError:
                if attempt == 3:
                    raise
                import os as _os
                import time as _time

                from .local_config import get_storage_config

                remount = get_storage_config().remount_command_for(path)
                if remount:
                    _os.system(remount)
                _time.sleep(5 * (2 ** attempt))
        # Qwen3-VL-class configs nest the LLM under text_config; lift it and
        # carry the vision/token-id extras alongside.
        vision_config = raw.get("vision_config")
        video_min_pixels = 4_096
        video_max_pixels = 25_165_824
        if vision_config is not None and raw.get("model_type", "").startswith("qwen3_vl"):
            video_processor_path = model_dir / "video_preprocessor_config.json"
            if video_processor_path.exists():
                video_processor = json.loads(video_processor_path.read_text())
                if not isinstance(video_processor, dict):
                    raise ValueError(
                        "video_preprocessor_config.json must contain an object")
                size = video_processor.get("size", {})
                if not isinstance(size, dict):
                    raise ValueError(
                        "video_preprocessor_config.json size must contain an object")
                for processor_key, vision_key in (
                    ("patch_size", "patch_size"),
                    ("temporal_patch_size", "temporal_patch_size"),
                    ("merge_size", "spatial_merge_size"),
                ):
                    processor_value = video_processor.get(processor_key)
                    if processor_value is None:
                        continue
                    if (not isinstance(processor_value, int)
                            or isinstance(processor_value, bool)
                            or processor_value <= 0):
                        raise ValueError(
                            f"video {processor_key} must be a positive integer")
                    if processor_value != vision_config.get(vision_key):
                        raise ValueError(
                            f"video {processor_key}={processor_value} does not "
                            f"match vision_config {vision_key}="
                            f"{vision_config.get(vision_key)!r}")
                video_min_pixels = _validated_pixel_bound(
                    size.get("shortest_edge"), video_min_pixels,
                    "video shortest_edge")
                video_max_pixels = _validated_pixel_bound(
                    size.get("longest_edge"), video_max_pixels,
                    "video longest_edge")
                if video_max_pixels < video_min_pixels:
                    raise ValueError(
                        "video longest_edge must be >= video shortest_edge")
        if "text_config" in raw and raw.get("model_type", "").startswith("qwen3_vl"):
            t = dict(raw["text_config"])
            t["model_type"] = raw["model_type"]
            t.setdefault("tie_word_embeddings", raw.get("tie_word_embeddings", False))
            for k in ("image_token_id", "video_token_id",
                      "vision_start_token_id", "vision_end_token_id"):
                if k in raw:
                    t[k] = raw[k]
            raw = t
        elif "text_config" in raw and raw.get("model_type", "") == "kimi_k25":
            # F93: Kimi K2.5's language model (text_config.model_type="kimi_k2")
            # is DeepSeek-style MLA+MoE, same field names GLM-5.2 already uses
            # (q_lora_rank, kv_lora_rank, n_routed_experts, ...) -- only the
            # nesting differs. Preserve the OUTER "kimi_k25" as model_type so
            # engine.py can dispatch on it distinctly from bare "kimi_k2".
            # Checkpoint tensor names are additionally prefixed
            # "language_model.model.layers.N...." (vision wrapper) instead of
            # "model.layers.N...." -- a WeightStore/loader-side concern, not
            # handled here.
            t = dict(raw["text_config"])
            t["model_type"] = raw["model_type"]
            t.setdefault("tie_word_embeddings", raw.get("tie_word_embeddings", False))
            raw = t

        vocab_size = raw["vocab_size"]
        eos = list(_validated_token_ids(
            raw.get("eos_token_id", []), vocab_size,
            "config.json eos_token_id"))
        generation_path = model_dir / "generation_config.json"
        if generation_path.exists():
            generation = json.loads(generation_path.read_text())
            if not isinstance(generation, dict):
                raise ValueError("generation_config.json must contain an object")
            for token_id in _validated_token_ids(
                    generation.get("eos_token_id"), vocab_size,
                    "generation_config.json eos_token_id"):
                if token_id not in eos:
                    eos.append(token_id)
        n_heads = raw["num_attention_heads"]

        linear_attn_config = raw.get("linear_attn_config")
        kda_layers: tuple[int, ...] = ()
        full_attn_layers: tuple[int, ...] = ()
        kda_head_dim = 0
        kda_num_heads = 0
        kda_conv_kernel_size = 4
        if linear_attn_config is not None:
            # F92: config.json lists are 1-indexed layer numbers; the rest of
            # this codebase (mlp_layer_types, indexer_types, ...) is 0-indexed.
            kda_layers = tuple(sorted(
                entry - 1 for entry in linear_attn_config.get("kda_layers", ())))
            full_attn_layers = tuple(sorted(
                entry - 1 for entry in linear_attn_config.get("full_attn_layers", ())))
            kda_head_dim = linear_attn_config.get("head_dim", 0)
            kda_num_heads = linear_attn_config.get("num_heads", 0)
            kda_conv_kernel_size = linear_attn_config.get("short_conv_kernel_size", 4)

        return cls(
            model_type=raw.get("model_type", "llama"),
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=n_heads,
            num_key_value_heads=raw.get("num_key_value_heads", n_heads),
            moe_intermediate_size=raw.get("moe_intermediate_size", 0),
            vocab_size=vocab_size,
            rms_norm_eps=raw.get("rms_norm_eps", 1e-5),
            rope_theta=raw.get("rope_theta") or raw.get("rope_parameters", {}).get("rope_theta", 10000.0),
            max_position_embeddings=raw.get("max_position_embeddings", 4096),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            attention_bias=raw.get("attention_bias", False),
            head_dim=raw.get("head_dim") or raw["hidden_size"] // n_heads,
            eos_token_ids=tuple(eos),
            torch_dtype=raw.get("torch_dtype", raw.get("dtype", "bfloat16")),
            num_experts=raw.get("num_experts", raw.get("n_routed_experts", raw.get("num_local_experts", 0))),
            num_experts_per_tok=raw.get("num_experts_per_tok", raw.get(
                "experts_per_token", raw.get("num_experts_per_token", 0))),
            norm_topk_prob=raw.get("norm_topk_prob", raw.get("moe_renormalize", False)),
            layer_types=tuple(raw.get("layer_types", ())),
            sliding_window=raw.get("sliding_window") or 0,
            swiglu_limit=raw.get("swiglu_limit", 7.0),
            rope_scaling=raw.get("rope_scaling"),
            first_k_dense_replace=raw.get("first_k_dense_replace", 0),
            qk_nope_head_dim=raw.get("qk_nope_head_dim", 0),
            qk_rope_head_dim=raw.get("qk_rope_head_dim", 0),
            v_head_dim=raw.get("v_head_dim", 0),
            # F92: Kimi Linear's config.json has an explicit "q_lora_rank": null
            # (no Q compression) -- raw.get(key, default) returns the JSON null,
            # not the default, when the key is present. Coerce None -> 0.
            q_lora_rank=raw.get("q_lora_rank") or 0,
            kv_lora_rank=raw.get("kv_lora_rank", 0),
            n_shared_experts=raw.get("n_shared_experts", raw.get("num_shared_experts", 0)),
            n_group=raw.get("n_group", raw.get("num_expert_group", 1)),
            topk_group=raw.get("topk_group", 1),
            routed_scaling_factor=raw.get("routed_scaling_factor", 1.0),
            mlp_layer_types=tuple(raw.get("mlp_layer_types", ())),
            num_nextn_predict_layers=raw.get("num_nextn_predict_layers", 0),
            rope_interleave=raw.get("rope_interleave", False),
            mla_latent_norm_eps=raw.get("mla_latent_norm_eps", 1e-6),
            vision_config=vision_config,
            image_token_id=raw.get("image_token_id", 0),
            video_token_id=raw.get("video_token_id", 0),
            vision_start_token_id=raw.get("vision_start_token_id", 0),
            vision_end_token_id=raw.get("vision_end_token_id", 0),
            video_min_pixels=video_min_pixels,
            video_max_pixels=video_max_pixels,
            index_topk=raw.get("index_topk", 0),
            index_head_dim=raw.get("index_head_dim", 128),
            index_n_heads=raw.get("index_n_heads", 32),
            indexer_types=tuple(raw.get("indexer_types", ())),
            index_topk_freq=raw.get("index_topk_freq", 1),
            index_skip_topk_offset=raw.get("index_skip_topk_offset", 2),
            index_share_for_mtp_iteration=raw.get("index_share_for_mtp_iteration", False),
            kda_layers=kda_layers,
            full_attn_layers=full_attn_layers,
            kda_head_dim=kda_head_dim,
            kda_num_heads=kda_num_heads,
            kda_conv_kernel_size=kda_conv_kernel_size,
            moe_layer_freq=raw.get("moe_layer_freq", 1),
            mla_use_nope=raw.get("mla_use_nope", False),
            moe_expert_prefix=(
                "block_sparse_moe.experts"
                if raw.get("model_type") in ("kimi_linear", "kimi_k25", "kimi_k2")
                else "mlp.experts"
            ),
        )


def validate_expert_top_k_by_layer(
    cfg: ModelConfig, schedule,
) -> tuple[int, ...]:
    """Canonicalize and validate an explicit layer-complete pruning schedule."""
    if not isinstance(schedule, (list, tuple)):
        raise ValueError(
            "expert_top_k_by_layer must be a list or tuple of integers")
    schedule = tuple(schedule)
    if not schedule:
        return ()
    if len(schedule) != cfg.num_hidden_layers:
        raise ValueError(
            "expert_top_k_by_layer must contain exactly "
            f"{cfg.num_hidden_layers} entries, got {len(schedule)}")
    for layer, top_k in enumerate(schedule):
        if (not isinstance(top_k, int) or isinstance(top_k, bool)
                or not 1 <= top_k <= cfg.num_experts_per_tok
                or top_k > cfg.num_experts):
            limit = min(cfg.num_experts_per_tok, cfg.num_experts)
            raise ValueError(
                "expert_top_k_by_layer entries must be integers within "
                f"[1, {limit}], got {top_k!r} at layer {layer}")
    return schedule


def effective_expert_top_k(cfg: ModelConfig, layer: int) -> int:
    """Return the active OLMoE expert count for ``layer``.

    An empty per-layer schedule preserves the checkpoint's released top-k.  A
    nonempty schedule is deliberately strict: it must cover every layer and may
    only retain or prune experts.  Failing closed here prevents a malformed
    lossy experiment from silently broadening back to the global top-k.
    """
    num_layers = cfg.num_hidden_layers
    if not isinstance(layer, int) or isinstance(layer, bool):
        raise TypeError("expert-routing layer must be an integer")
    if not 0 <= layer < num_layers:
        raise IndexError(
            f"expert-routing layer {layer} is outside [0, {num_layers})")

    released_top_k = cfg.num_experts_per_tok
    schedule = getattr(cfg, "expert_top_k_by_layer", ())
    if schedule:
        if len(schedule) != num_layers:
            raise ValueError(
                "expert_top_k_by_layer must contain exactly "
                f"{num_layers} entries, got {len(schedule)}")
        top_k = schedule[layer]
        if (not isinstance(top_k, int) or isinstance(top_k, bool)
                or not 1 <= top_k <= released_top_k):
            raise ValueError(
                "expert_top_k_by_layer entries must be integers within "
                f"[1, {released_top_k}], got {top_k!r} at layer {layer}")
    else:
        top_k = released_top_k

    if (not isinstance(top_k, int) or isinstance(top_k, bool)
            or not 1 <= top_k <= cfg.num_experts):
        raise ValueError(
            "effective expert top-k must be an integer within "
            f"[1, {cfg.num_experts}], got {top_k!r} at layer {layer}")
    return top_k
