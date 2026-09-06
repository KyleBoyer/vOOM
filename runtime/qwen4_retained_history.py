"""Exact storage compaction for an opt-in, complete Qwen4 prefix fork.

No tensor imports at module load. This is an ownership operation, not a model
operator: retain independent 16-bit convolution tails rather than their large
padded prefill inputs. FP32 recurrent matrices and attention state are untouched.
"""

import time


def retained_qsa_backing_allowance(cfg, *, prefix_tokens, tile_tokens):
    """Conservative raw-key view backing for one projected indexer tile.

    The retained KV projection already charges the one raw index-key head;
    its slice can additionally own all query heads from that same tile. This
    allowance is not measured resident memory or a bound on all lazy graphs.
    """
    return (min(prefix_tokens, tile_tokens)
            * sum(kind == "full_attention" for kind in cfg.layer_types)
            * cfg.qwen4_indexer_n_heads * cfg.qwen4_indexer_head_dim * 2)


def copy_history_bits(value):
    """Materialize independent BF16/F16 payloads without floating-point casts."""
    import mlx.core as mx
    import numpy as np

    if value.dtype not in (mx.bfloat16, mx.float16):
        raise ValueError("retained convolution history must have a 16-bit dtype")
    host = np.asarray(value.view(mx.uint16)).copy(order="C")
    result = mx.array(host, dtype=mx.uint16).view(value.dtype)
    mx.eval(result)
    return result


def compact_retained_history(fork, endpoint, cfg, *, prefix_tokens, kv_type,
                             expected_dtype, copy_bits, reserve,
                             active_bytes, clock=time.perf_counter):
    """Compact a fresh complete RAM fork before the original advances.

    Validate all geometry/owners before reserving or copying. Stage every small
    replacement before publishing any of them, so a validation/copy/reservation
    failure leaves the fork unchanged. Both owners still share the old arrays
    at entry; physical backing may become reclaimable only as the ORIGINAL
    endpoint advances. The caller must not call immediate active-byte change a
    prediction of eventual release.
    """
    if (type(fork) is not kv_type or type(endpoint) is not kv_type or fork is endpoint
            or cfg.model_type != "qwen4_exp" or type(prefix_tokens) is not int
            or prefix_tokens <= 0 or fork.offset != prefix_tokens
            or endpoint.offset != prefix_tokens
            or fork.compressed_mla or endpoint.compressed_mla):
        raise ValueError("compaction requires a separate complete plain Qwen4 prefix fork")
    count = cfg.num_hidden_layers
    layer_types = tuple(cfg.layer_types)
    if (len(layer_types) != count
            or any(kind not in ("full_attention", "linear_attention") for kind in layer_types)):
        raise ValueError("unsupported Qwen4 layer layout")
    kda, original_kda = fork.kda_cache, endpoint.kda_cache
    aux, original_aux = fork.qwen4_cache, endpoint.qwen4_cache
    if kda is original_kda or aux is original_aux:
        raise ValueError("retained companions must have independent owners")
    for owner in (kda, original_kda):
        if (getattr(owner, "spill_enabled", False) or getattr(owner, "_spill_meta", None)
                or getattr(owner, "_factor_capture", None) is not None):
            raise ValueError("retained compaction requires resident non-factor state")
    # Every mutable model-state list is independently owned by the fresh fork.
    for left, right, fields in (
        (fork, endpoint, ("keys", "values", "_starts", "_windows")),
        (kda, original_kda, ("_state", "_conv")),
        (aux, original_aux, ("qsa_keys", "qsa_positions", "qsa_pooled_keys",
                            "ple_conv", "ple_context", "ple_lengths")),
    ):
        for field in fields:
            a, b = getattr(left, field), getattr(right, field)
            if (not isinstance(a, list) or not isinstance(b, list)
                    or a is b or len(a) != count or len(b) != count):
                raise ValueError("retained state requires complete independent lists")
    if aux.qsa_pool_cache_enabled != original_aux.qsa_pool_cache_enabled:
        raise ValueError("retained QSA policy mismatch")
    ple_layers = set(cfg.qwen4_ple_layers)
    if any(type(layer) is not int or not 0 <= layer < count for layer in ple_layers):
        raise ValueError("invalid PLE layer indices")
    conv_width = (2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
                  + cfg.linear_num_value_heads * cfg.linear_value_head_dim)
    kda_shape = (1, max(0, cfg.linear_conv_kernel_dim - 1), conv_width)
    ple_shape = (1, max(0, cfg.qwen4_ple_conv_kernel_size - 1) * cfg.qwen4_ngram_size,
                 cfg.qwen4_hc_count * cfg.hidden_size)
    candidates = []
    for layer, kind in enumerate(layer_types):
        if (fork._starts[layer] != endpoint._starts[layer]
                or fork._windows[layer] != endpoint._windows[layer]):
            raise ValueError("retained KV position metadata mismatch")
        for field in ("keys", "values"):
            if getattr(fork, field)[layer] is not getattr(endpoint, field)[layer]:
                raise ValueError("compaction requires a fresh shared-array fork")
        if kind == "full_attention":
            if any(value is None or fork._starts[layer] + int(value.shape[2]) != prefix_tokens
                   for value in (fork.keys[layer], fork.values[layer])):
                raise ValueError("retained full-attention layer has incomplete prefix")
            if (aux.qsa_keys[layer] is None or aux.qsa_positions[layer] is None
                    or tuple(aux.qsa_keys[layer].shape) != (1, prefix_tokens, cfg.qwen4_indexer_head_dim)
                    or tuple(aux.qsa_positions[layer].shape) != (1, prefix_tokens)
                    or kda._state[layer] is not None):
                raise ValueError("retained full-attention auxiliary state is incomplete")
        elif any(value is not None for value in (
                fork.keys[layer], fork.values[layer], aux.qsa_keys[layer],
                aux.qsa_positions[layer], aux.qsa_pooled_keys[layer])):
            raise ValueError("unexpected attention state on retained DeltaNet layer")
        for field in ("qsa_keys", "qsa_positions", "qsa_pooled_keys", "ple_conv"):
            if getattr(aux, field)[layer] is not getattr(original_aux, field)[layer]:
                raise ValueError("compaction requires fresh auxiliary arrays")
        if (kda._state[layer] is not original_kda._state[layer]
                or aux.ple_context[layer] != original_aux.ple_context[layer]
                or aux.ple_lengths[layer] != original_aux.ple_lengths[layer]):
            raise ValueError("retained recurrent endpoint mismatch")
        history, original = kda._conv[layer], original_kda._conv[layer]
        if kind == "linear_attention":
            if (not isinstance(history, tuple) or not isinstance(original, tuple)
                    or len(history) != 1 or len(original) != 1
                    or history[0] is None or history[0] is not original[0]
                    or kda._state[layer] is None):
                raise ValueError("incomplete retained DeltaNet history")
            candidates.append(("kda", layer, history[0], kda_shape))
        elif history is not None or original is not None:
            raise ValueError("unexpected full-attention convolution history")
        if layer in ple_layers:
            if (aux.ple_conv[layer] is None or aux.ple_lengths[layer] != prefix_tokens
                    or not isinstance(aux.ple_context[layer], tuple)
                    or len(aux.ple_context[layer]) != cfg.qwen4_ngram_size - 1):
                raise ValueError("incomplete retained PLE history")
            candidates.append(("ple", layer, aux.ple_conv[layer], ple_shape))
        elif (aux.ple_conv[layer] is not None or aux.ple_lengths[layer] != 0
              or aux.ple_context[layer] is not None):
            raise ValueError("unexpected retained PLE history")
    for _, _, value, shape in candidates:
        if tuple(value.shape) != shape or value.dtype != expected_dtype:
            raise ValueError("retained convolution shape/dtype mismatch")
    if not candidates:
        raise ValueError("retained compaction requires convolution histories")
    payload = sum(int(value.nbytes) for _, _, value, _ in candidates)
    # Full replacement set plus conservative host-copy scratch. The old
    # resident originals are already live; do not credit future release here.
    scratch = 2 * payload
    reserve(scratch)
    started = clock()
    before = int(active_bytes())
    replacements = []
    for kind, layer, value, _ in candidates:
        replacement = copy_bits(value)
        if (replacement is value or replacement.shape != value.shape
                or replacement.dtype != value.dtype):
            raise ValueError("retained history copy must be independent and same-format")
        replacements.append((kind, layer, replacement))
    for kind, layer, replacement in replacements:
        if kind == "kda":
            kda._conv[layer] = (replacement,)
        else:
            aux.ple_conv[layer] = replacement
    return {
        "qwen4_retained_conv_compact_calls": 1,
        "qwen4_retained_conv_compact_arrays": len(candidates),
        "qwen4_retained_conv_compact_bytes": payload,
        "qwen4_retained_conv_compact_scratch_bytes": scratch,
        "qwen4_retained_conv_compact_prefix_tokens": prefix_tokens,
        "qwen4_retained_conv_compact_active_before_bytes": before,
        "qwen4_retained_conv_compact_active_after_bytes": int(active_bytes()),
        "qwen4_retained_conv_compact_seconds": clock() - started,
    }
