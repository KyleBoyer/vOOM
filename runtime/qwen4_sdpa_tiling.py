"""Explicit, unconnected Qwen head-256 SDPA scheduling candidate.

No runtime flag/default, MLX import, state mutation, cache/policy change or
claim of numerical equivalence. Query-shape changes require a bitwise gate.
The caller owns admission for full KV/mask and bounded score scratch.
"""


def query_tiles(total, tile_queries):
    if type(total) is not int or not 1 <= total <= 1_048_576:
        raise ValueError('positive bounded query count required')
    if type(tile_queries) is not int or tile_queries not in (0, 128, 256, 512):
        raise ValueError('query tile must be 0, 128, 256 or 512')
    if not tile_queries or total <= tile_queries:
        return ((0, total),)
    ranges, start = [], 0
    while start < total:
        end = min(start + tile_queries, total)
        # Avoid introducing a <=8-row tail and changing to the vector SDPA
        # dispatch family. The largest resulting tile is tile_queries+8.
        if 0 < total - end <= 8:
            end = total
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def query_tiled_sdpa(mx, query, keys, values, *, scale, mask, tile_queries):
    """Split only query rows of an already-built attention operation.

    Full keys/values and the exact precomputed mask are shared unchanged.
    Each partial output is evaluated before constructing the next. No host
    copies, allocator clearing, peak reset, projection or state update occurs.
    None/array masks only: a causal-string mask would change sliced offsets.
    """
    ranges = query_tiles(int(query.shape[2]), tile_queries)
    if len(ranges) == 1:
        return mx.fast.scaled_dot_product_attention(query, keys, values,
                                                    scale=scale, mask=mask)
    if (query.ndim != 4 or keys.ndim != 4 or values.ndim != 4
            or query.shape[0] != keys.shape[0] or keys.shape[:3] != values.shape[:3]
            or query.shape[-1] != 256 or keys.shape[-1] != 256 or values.shape[-1] != 256
            or query.shape[1] % keys.shape[1]
            or query.dtype != keys.dtype or query.dtype != values.dtype
            or query.dtype not in (mx.bfloat16, mx.float16)):
        raise ValueError('head-256 FP16/BF16 Qwen-compatible geometry required')
    if mask is not None and (getattr(mask, 'ndim', None) != 4
            or mask.shape[0] not in (1, query.shape[0])
            or mask.shape[1] not in (1, query.shape[1])
            or mask.shape[2] not in (1, query.shape[2])
            or mask.shape[3] not in (1, keys.shape[2])):
        raise ValueError('precomputed broadcast-compatible rank-four mask required')
    outputs = []
    for start, end in ranges:
        part_mask = (mask if mask is None or mask.shape[2] == 1
                     else mask[:, :, start:end, :])
        output = mx.fast.scaled_dot_product_attention(
            query[:, :, start:end, :], keys, values, scale=scale, mask=part_mask)
        mx.eval(output)
        outputs.append(output)
    return mx.concatenate(outputs, axis=2)
