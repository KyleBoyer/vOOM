"""Metal decoder for exact block-decodable BF16 NF12 sidecars."""

from __future__ import annotations

from functools import cache

import mlx.core as mx
import numpy as np

from formats.bf16_nf12_sidecar import (
    BLOCK_HEADER_BYTES,
    BLOCK_VALUES,
    CODE_BYTES_PER_BLOCK,
    PATCH_BYTES,
)

BLOCKS_PER_THREADGROUP = 16


@cache
def _decoder_kernel():
    source = f"""
        constexpr uint BLOCK_VALUES = {BLOCK_VALUES};
        constexpr uint HEADER_BYTES = {BLOCK_HEADER_BYTES};
        constexpr uint CODE_BYTES = {CODE_BYTES_PER_BLOCK};
        constexpr uint PATCH_BYTES = {PATCH_BYTES};
        constexpr uint BLOCKS_PER_TG = {BLOCKS_PER_THREADGROUP};

        uint local = thread_index_in_threadgroup;
        uint group = threadgroup_position_in_grid.x;
        uint local_first_block = group * BLOCKS_PER_TG;
        uint blocks = selected_block_count;
        uint codes_base = header_stream_bytes;

        for (uint within = 0; within < BLOCKS_PER_TG; ++within) {{
            uint selected_block = local_first_block + within;
            if (selected_block >= blocks) return;
            uint block = first_block + selected_block;
            uint header = block * HEADER_BYTES;
            uint patch_offset =
                uint(encoded[header]) |
                (uint(encoded[header + 1]) << 8) |
                (uint(encoded[header + 2]) << 16) |
                (uint(encoded[header + 3]) << 24);
            uint output_offset =
                uint(encoded[header + 4]) |
                (uint(encoded[header + 5]) << 8) |
                (uint(encoded[header + 6]) << 16) |
                (uint(encoded[header + 7]) << 24);
            uint valid =
                uint(encoded[header + 8]) |
                (uint(encoded[header + 9]) << 8);
            uint exception_count =
                uint(encoded[header + 10]) |
                (uint(encoded[header + 11]) << 8);
            ushort mode = ushort(encoded[header + 12] & 15);
            if (local >= valid) continue;

            uint pair = local >> 1;
            uint code_offset =
                codes_base + block * CODE_BYTES + pair * 3;
            ushort code;
            if ((local & 1) == 0) {{
                code =
                    ushort(encoded[code_offset]) |
                    (ushort(encoded[code_offset + 1] & 15) << 8);
            }} else {{
                code =
                    ushort(encoded[code_offset + 1] >> 4) |
                    (ushort(encoded[code_offset + 2]) << 4);
            }}
            ushort value =
                (code & ushort(0x07ff)) |
                ((code & ushort(0x0800)) << 4) |
                (mode << 11);
            for (uint patch = 0; patch < exception_count; ++patch) {{
                uint at = patch_offset + patch * PATCH_BYTES;
                if (uint(encoded[at]) == local) {{
                    value =
                        ushort(encoded[at + 1]) |
                        (ushort(encoded[at + 2]) << 8);
                    break;
                }}
            }}
            out[output_offset - output_base + local] = value;
        }}
    """
    return mx.fast.metal_kernel(
        name="voom_bf16_nf12_decode_v1",
        input_names=[
            "encoded", "first_block", "selected_block_count",
            "header_stream_bytes", "output_base",
        ],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=False,
    )


def _validated_encoded(
    encoded_raw: bytes | mx.array, entry: dict
) -> mx.array:
    """Validate and normalize one immutable encoded layer stream."""
    expected = int(entry["file_bytes"])
    actual = (
        int(encoded_raw.size)
        if isinstance(encoded_raw, mx.array)
        else len(encoded_raw)
    )
    if actual != expected:
        raise ValueError(
            f"BF16 NF12 encoded bytes {actual} != {expected}"
        )
    encoded = (
        encoded_raw
        if isinstance(encoded_raw, mx.array)
        else mx.array(np.frombuffer(encoded_raw, dtype=np.uint8))
    )
    if encoded.dtype != mx.uint8:
        raise ValueError("BF16 NF12 encoded tensor must be uint8")
    return encoded


def _decode_span(
    encoded: mx.array,
    entry: dict,
    *,
    first_block: int,
    block_count: int,
    output_base: int,
    output_value_count: int,
) -> mx.array:
    if block_count <= 0 or output_value_count <= 0:
        raise ValueError("BF16 NF12 decode span must be non-empty")
    total_blocks = int(entry["block_count"])
    header_bytes = int(entry["header_bytes"])
    if header_bytes != total_blocks * BLOCK_HEADER_BYTES:
        raise ValueError("BF16 NF12 header/block count mismatch")
    if first_block < 0 or first_block + block_count > total_blocks:
        raise ValueError("BF16 NF12 decode span exceeds layer blocks")
    group_count = (
        block_count + BLOCKS_PER_THREADGROUP - 1
    ) // BLOCKS_PER_THREADGROUP
    decoded_bits = _decoder_kernel()(
        inputs=[
            encoded,
            mx.array(first_block, dtype=mx.uint32),
            mx.array(block_count, dtype=mx.uint32),
            mx.array(header_bytes, dtype=mx.uint32),
            mx.array(output_base, dtype=mx.uint32),
        ],
        grid=(group_count * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(output_value_count,)],
        output_dtypes=[mx.uint16],
    )[0]
    return decoded_bits.view(mx.bfloat16)


def decode_layer(
    encoded_raw: bytes | mx.array, entry: dict
) -> dict[str, mx.array]:
    """Decode one complete layer in one dispatch and return BF16 views."""
    encoded = _validated_encoded(encoded_raw, entry)
    decoded = _decode_span(
        encoded,
        entry,
        first_block=0,
        block_count=int(entry["block_count"]),
        output_base=0,
        output_value_count=int(entry["output_value_count"]),
    )
    output = {
        tensor["name"]: decoded[
            int(tensor["output_offset"]) :
            int(tensor["output_offset"]) + int(tensor["value_count"])
        ].reshape(tuple(int(value) for value in tensor["shape"]))
        for tensor in entry["tensors"]
    }
    mx.eval(list(output.values()))
    return output


def decode_names(
    encoded_raw: bytes | mx.array,
    entry: dict,
    requested_names: list[str] | tuple[str, ...],
) -> dict[str, mx.array]:
    """Decode only requested tensors, one exact block span per tensor."""
    encoded = _validated_encoded(encoded_raw, entry)
    specs = {tensor["name"]: tensor for tensor in entry["tensors"]}
    names = list(dict.fromkeys(requested_names))
    tensors = []
    for name in names:
        tensor = specs.get(name)
        if tensor is None:
            raise KeyError(f"{name}: tensor is not encoded in NF12 layer")
        tensors.append(tensor)
    physical_order = sorted(
        tensors, key=lambda tensor: int(tensor["block_start"])
    )
    runs: list[list[dict]] = []
    for tensor in physical_order:
        if runs:
            previous = runs[-1][-1]
            contiguous = (
                int(tensor["block_start"])
                == int(previous["block_start"])
                + int(previous["block_count"])
                and int(tensor["output_offset"])
                == int(previous["output_offset"])
                + int(previous["value_count"])
            )
        else:
            contiguous = False
        if contiguous:
            runs[-1].append(tensor)
        else:
            runs.append([tensor])

    decoded_by_name = {}
    for run in runs:
        output_base = int(run[0]["output_offset"])
        decoded = _decode_span(
            encoded,
            entry,
            first_block=int(run[0]["block_start"]),
            block_count=sum(
                int(tensor["block_count"])
                for tensor in run
            ),
            output_base=output_base,
            output_value_count=sum(
                int(tensor["value_count"])
                for tensor in run
            ),
        )
        for tensor in run:
            output_start = int(tensor["output_offset"]) - output_base
            value_count = int(tensor["value_count"])
            decoded_by_name[tensor["name"]] = decoded[
                output_start : output_start + value_count
            ].reshape(
                tuple(int(dimension) for dimension in tensor["shape"])
            )
    output = {name: decoded_by_name[name] for name in names}
    mx.eval(list(output.values()))
    return output
