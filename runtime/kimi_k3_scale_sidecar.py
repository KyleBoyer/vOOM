"""Metal materialization for exact Kimi K3 E8M0 scale sidecars."""

from __future__ import annotations

from functools import cache

import mlx.core as mx
import numpy as np

from formats.kimi_k3_scale_sidecar import (
    HEADER_BYTES_PER_EXPERT,
    PROJECTIONS,
    ScaleRecord,
    assemble_decode_batch,
)


@cache
def _decoder_kernel(scale_count: int):
    source = f"""
        constexpr uint SCALE_COUNT = {scale_count};
        constexpr uint HEADER_BYTES = {HEADER_BYTES_PER_EXPERT};
        constexpr uint VALUES_PER_THREAD = 16;
        constexpr uint BLOCKS_PER_TENSOR =
            (SCALE_COUNT + VALUES_PER_THREAD - 1) / VALUES_PER_THREAD;
        uint block = thread_position_in_grid.x;
        uint tensor = block / BLOCKS_PER_TENSOR;
        uint block_in_tensor = block - tensor * BLOCKS_PER_TENSOR;
        uint element0 = block_in_tensor * VALUES_PER_THREAD;
        uint expert = tensor / 3;
        uint projection = tensor - expert * 3;
        uint header = expert * HEADER_BYTES;

        uint record_offset =
            uint(sidecar[header]) |
            (uint(sidecar[header + 1]) << 8) |
            (uint(sidecar[header + 2]) << 16) |
            (uint(sidecar[header + 3]) << 24);
        uchar base0 = sidecar[header + 4];
        uchar bits0 = sidecar[header + 5];
        uchar base1 = sidecar[header + 6];
        uchar bits1 = sidecar[header + 7];
        uchar base2 = sidecar[header + 8];
        uchar bits2 = sidecar[header + 9];

        uint payload_offset = record_offset;
        uchar base = base0;
        uchar bits = bits0;
        if (projection > 0) {{
            payload_offset += (SCALE_COUNT * uint(bits0) + 7) >> 3;
            base = base1;
            bits = bits1;
        }}
        if (projection > 1) {{
            payload_offset += (SCALE_COUNT * uint(bits1) + 7) >> 3;
            base = base2;
            bits = bits2;
        }}

        uint output_offset = tensor * SCALE_COUNT + element0;
        uint valid = min(VALUES_PER_THREAD, SCALE_COUNT - element0);
        if (bits == 0) {{
            for (uint i = 0; i < valid; ++i) {{
                out[output_offset + i] = base;
            }}
        }} else if (bits == 1) {{
            uint encoded_offset = payload_offset + (element0 >> 3);
            for (uint byte_index = 0; byte_index < 2; ++byte_index) {{
                uchar encoded = sidecar[encoded_offset + byte_index];
                uint first = byte_index * 8;
                for (uint bit = 0; bit < 8; ++bit) {{
                    if (first + bit < valid) {{
                        out[output_offset + first + bit] =
                            base + ((encoded >> bit) & 1);
                    }}
                }}
            }}
        }} else if (bits == 2) {{
            uint encoded_offset = payload_offset + (element0 >> 2);
            for (uint byte_index = 0; byte_index < 4; ++byte_index) {{
                uchar encoded = sidecar[encoded_offset + byte_index];
                uint first = byte_index * 4;
                if (first < valid) out[output_offset + first] =
                    base + (encoded & 3);
                if (first + 1 < valid) out[output_offset + first + 1] =
                    base + ((encoded >> 2) & 3);
                if (first + 2 < valid) out[output_offset + first + 2] =
                    base + ((encoded >> 4) & 3);
                if (first + 3 < valid) out[output_offset + first + 3] =
                    base + ((encoded >> 6) & 3);
            }}
        }} else if (bits == 4) {{
            uint encoded_offset = payload_offset + (element0 >> 1);
            for (uint byte_index = 0; byte_index < 8; ++byte_index) {{
                uchar encoded = sidecar[encoded_offset + byte_index];
                uint first = byte_index * 2;
                if (first < valid) out[output_offset + first] =
                    base + (encoded & 15);
                if (first + 1 < valid) out[output_offset + first + 1] =
                    base + (encoded >> 4);
            }}
        }} else {{
            uint encoded_offset = payload_offset + element0;
            for (uint i = 0; i < valid; ++i) {{
                out[output_offset + i] =
                    base + sidecar[encoded_offset + i];
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"voom_k3_scale_sidecar_decode_v2_n{scale_count}",
        input_names=["sidecar"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=False,
    )


def decode_records(
    records: list[ScaleRecord],
    projection_shapes: dict[str, tuple[int, ...]],
) -> dict[tuple[int, str], mx.array]:
    """Decode records in one dispatch and return original-shaped views."""
    if not records:
        return {}
    counts = {
        int(np.prod(projection_shapes[projection], dtype=np.int64))
        for projection in PROJECTIONS
    }
    if len(counts) != 1:
        raise ValueError(
            "fused K3 scale-sidecar decode needs equal projection counts"
        )
    scale_count = counts.pop()
    encoded_raw = assemble_decode_batch(records)
    encoded = mx.array(np.frombuffer(encoded_raw, dtype=np.uint8))
    blocks_per_tensor = (scale_count + 15) // 16
    output_blocks = len(records) * len(PROJECTIONS) * blocks_per_tensor
    decoded = _decoder_kernel(scale_count)(
        inputs=[encoded],
        grid=(output_blocks, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(len(records), len(PROJECTIONS), scale_count)],
        output_dtypes=[mx.uint8],
    )[0]
    output = {
        (record.expert, projection): decoded[local, projection_index].reshape(
            projection_shapes[projection]
        )
        for local, record in enumerate(records)
        for projection_index, projection in enumerate(PROJECTIONS)
    }
    # Fetch can execute on a prefetch worker.  Materialize every view on the
    # thread whose MLX stream created it before returning to the main thread.
    mx.eval(list(output.values()))
    return output
