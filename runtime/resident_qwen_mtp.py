"""Experimental native Qwen3.5 MTP for the fully-resident MLX-LM backend.

Qwen3.5 ships a one-layer multi-token-prediction (MTP) head.  MLX-LM 0.31.3
loads the trunk but deliberately drops ``mtp.*`` tensors, so the resident
backend cannot use the released draft head through MLX-LM's public model
object.  This module loads only that small head and supplies two operations:

* an exact trunk forward that also returns the released pre-final-norm hidden
  state consumed by MTP; and
* an MTP forward over the released head and the trunk's shared embedding and
  language-model projection.

The serving loop performs target verification and stochastic requests use an
independent target draw.  This path nevertheless remains opt-in and is not a
released-model-correct serving path: on the current all-MXFP4 checkpoint, a
two-position target sweep is not numerically identical to two one-position
sweeps and fails the greedy byte-identity gate.  Keep it as an instrumented
kernel experiment until a row-independent verifier passes that gate.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".pre_fc_norm_hidden.weight",
    ".pre_fc_norm_embedding.weight",
    "mtp.norm.weight",
)


def _confirmed_linear_attention(
    layer, inputs: mx.array, mask, cache, confirmed: int,
) -> mx.array:
    """Fold confirmed and speculative positions through DeltaNet in order.

    The installed MLX-LM 0.31.3 kernel accepts a multi-position decode input,
    but that fold is not byte-identical to two one-position recurrent updates.
    Keep the large input projections and following MLP batched while splitting
    only the stateful convolution/DeltaNet recurrence at the confirmation
    boundary.  This is the model-side rollback technique proposed upstream for
    Qwen3.5 native MTP.
    """
    from mlx_lm.models.gated_delta import gated_delta_update

    batch, sequence, _ = inputs.shape
    qkv = layer.in_proj_qkv(inputs)
    z = layer.in_proj_z(inputs).reshape(
        batch, sequence, layer.num_v_heads, layer.head_v_dim)
    b = layer.in_proj_b(inputs)
    a = layer.in_proj_a(inputs)
    conv_state = (
        cache[0] if cache is not None and cache[0] is not None
        else mx.zeros(
            (batch, layer.conv_kernel_size - 1, layer.conv_dim),
            dtype=inputs.dtype))
    ssm_state = cache[1] if cache is not None else None
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)

    def fold(qkv_part, a_part, b_part, conv, state, part_mask):
        part_length = qkv_part.shape[1]
        conv_input = mx.concatenate([conv, qkv_part], axis=1)
        keep = layer.conv_kernel_size - 1
        next_conv = mx.contiguous(conv_input[:, -keep:])
        convolved = nn.silu(layer.conv1d(conv_input))
        q, k, v = [
            value.reshape(batch, part_length, heads, dimension)
            for value, heads, dimension in zip(
                mx.split(
                    convolved, [layer.key_dim, 2 * layer.key_dim], -1),
                [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
                [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim],
                strict=True)
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        output, next_state = gated_delta_update(
            q, k, v, a_part, b_part, layer.A_log, layer.dt_bias,
            state, part_mask, use_kernel=not layer.training)
        return output, next_conv, next_state

    confirmed_mask = (
        mask[:, :confirmed] if mask is not None else None)
    draft_mask = (
        mask[:, confirmed:] if mask is not None else None)
    output_confirmed, conv_confirmed, state_confirmed = fold(
        qkv[:, :confirmed], a[:, :confirmed], b[:, :confirmed],
        conv_state, ssm_state, confirmed_mask)
    output_draft, conv_final, state_final = fold(
        qkv[:, confirmed:], a[:, confirmed:], b[:, confirmed:],
        conv_confirmed, state_confirmed, draft_mask)
    output = mx.concatenate([output_confirmed, output_draft], axis=1)
    if cache is not None:
        cache[0] = conv_final
        cache[1] = state_final
        cache.advance(sequence)
    output = layer.norm(output, z)
    return layer.out_proj(output.reshape(batch, sequence, -1))


class _MTPDecoderLayer(nn.Module):
    """The released MTP block is one ordinary full-attention Qwen layer."""

    def __init__(self, args):
        super().__init__()
        from mlx_lm.models.qwen3_5 import Attention, MLP

        self.self_attn = Attention(args)
        self.input_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x, mask=None, cache=None):
        residual = x
        x = residual + self.self_attn(
            self.input_layernorm(x), mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class _MTPHead(nn.Module):
    def __init__(self, args, layers: int):
        super().__init__()
        self.pre_fc_norm_hidden = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(
            args.hidden_size * 2, args.hidden_size, bias=False)
        self.layers = [_MTPDecoderLayer(args) for _ in range(layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, hidden, next_ids, embedding, cache):
        from mlx_lm.models.base import create_attention_mask

        embedded = embedding(next_ids)
        fused = self.fc(mx.concatenate([
            self.pre_fc_norm_embedding(embedded),
            self.pre_fc_norm_hidden(hidden),
        ], axis=-1))
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(fused, cache[0])
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            fused = layer(fused, mask=mask, cache=layer_cache)
        return self.norm(fused)


def _load_released_mtp_weights(
    model_dir: Path,
) -> tuple[dict[str, mx.array], int]:
    """Load only ``mtp.*`` tensors from the checkpoint's indexed shards."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError("native MTP requires model.safetensors.index.json")
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map", {})
    names = sorted(name for name in weight_map if name.startswith("mtp."))
    if not names:
        raise ValueError("checkpoint has no released mtp.* tensors")

    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)

    selected: dict[str, mx.array] = {}
    for shard, shard_names in by_shard.items():
        loaded = mx.load(str(model_dir / shard))
        try:
            for name in shard_names:
                value = loaded[name]
                # This vModel checkpoint preserves the released HF Qwen3.5
                # RMSNorm delta convention.  MLX nn.RMSNorm consumes the
                # effective scale, exactly as the installed trunk sanitizer
                # does for its own tensors.
                if any(name.endswith(suffix) for suffix in _NORM_SUFFIXES):
                    value = value + 1.0
                selected[name.removeprefix("mtp.")] = value
        finally:
            del loaded
    return selected, len(names)


class ResidentQwenMTP:
    """Small released MTP head sharing the resident trunk's embed/lm_head."""

    cache_bytes_per_token = 2 * 4 * 256 * 2

    def __init__(self, model, model_dir: str | Path, quantization: dict):
        if not hasattr(model, "language_model"):
            raise ValueError("native MTP currently requires Qwen3.5 text model")
        language_model = model.language_model
        args = language_model.args
        raw_config = json.loads(
            (Path(model_dir) / "config.json").read_text())
        text_config = raw_config.get("text_config", raw_config)
        layers = int(text_config.get(
            "mtp_num_hidden_layers",
            getattr(args, "mtp_num_hidden_layers", 0)) or 0)
        if layers <= 0:
            raise ValueError("Qwen config does not declare an MTP layer")
        if int(getattr(args, "num_experts", 0) or 0):
            raise ValueError("resident native MTP currently supports dense Qwen")

        started = time.perf_counter()
        weights, tensor_count = _load_released_mtp_weights(Path(model_dir))
        head = _MTPHead(args, layers)
        group_size = int(quantization.get("group_size", 32))
        bits = int(quantization.get("bits", 4))
        mode = str(quantization.get("mode", "mxfp4"))

        def quantize_if_present(path, module):
            return (
                hasattr(module, "to_quantized")
                and f"{path}.scales" in weights)

        nn.quantize(
            head, group_size=group_size, bits=bits, mode=mode,
            class_predicate=quantize_if_present)
        head.load_weights(list(weights.items()), strict=True)
        head.eval()
        mx.eval(head.parameters())
        del weights
        gc.collect()

        self.model = model
        self.language_model = language_model
        self.head = head
        self.tensor_count = tensor_count
        self.load_s = time.perf_counter() - started
        self.active_bytes = sum(
            int(value.nbytes)
            for value in _walk_arrays(self.head.parameters()))

    def make_cache(self):
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in self.head.layers]

    def trunk_forward(
        self, inputs: mx.array, cache, *, confirmed_prefix: int = 0,
    ):
        """Run the installed Qwen trunk and expose its pre-final-norm hidden."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        language_model = self.language_model
        core = language_model.model
        hidden = core.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(core.layers)
        full_attention_mask = create_attention_mask(
            hidden, cache[core.fa_idx])
        recurrent_mask = create_ssm_mask(hidden, cache[core.ssm_idx])
        for layer, layer_cache in zip(core.layers, cache, strict=True):
            mask = recurrent_mask if layer.is_linear else full_attention_mask
            if (layer.is_linear and confirmed_prefix > 0
                    and confirmed_prefix < hidden.shape[1]):
                residual = hidden
                attention = _confirmed_linear_attention(
                    layer.linear_attn, layer.input_layernorm(hidden),
                    mask, layer_cache, confirmed_prefix)
                after_attention = residual + attention
                hidden = after_attention + layer.mlp(
                    layer.post_attention_layernorm(after_attention))
            else:
                hidden = layer(hidden, mask=mask, cache=layer_cache)
        normalized = core.norm(hidden)
        if language_model.args.tie_word_embeddings:
            logits = core.embed_tokens.as_linear(normalized)
        else:
            logits = language_model.lm_head(normalized)
        return logits, hidden

    def draft_logits(self, hidden: mx.array, next_ids: mx.array, cache):
        hidden = self.advance(hidden, next_ids, cache)
        if self.language_model.args.tie_word_embeddings:
            return self.language_model.model.embed_tokens.as_linear(hidden)
        return self.language_model.lm_head(hidden)

    def advance(self, hidden: mx.array, next_ids: mx.array, cache):
        """Advance MTP attention state without materializing vocab logits."""
        return self.head(
            hidden, next_ids, self.language_model.model.embed_tokens, cache)

    def close(self):
        head = getattr(self, "head", None)
        if head is not None:
            self.head = None
            del head
        gc.collect()
        mx.clear_cache()


def _walk_arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_arrays(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_arrays(item)
