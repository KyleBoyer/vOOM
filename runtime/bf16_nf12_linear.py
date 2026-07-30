"""Small-M linear kernels that consume exact compact NF12 operands directly."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import mlx.core as mx

from formats.bf16_nf12_sidecar import (
    BLOCK_HEADER_BYTES,
    BLOCK_VALUES,
    CODE_BYTES_PER_BLOCK,
    PATCH_BYTES,
)


def direct_linear_eligible(spec: dict) -> bool:
    shape = tuple(int(value) for value in spec["shape"])
    return (
        len(shape) == 2
        and shape[0] % 8 == 0
        and shape[1] % 512 == 0
        and int(spec["raw_bytes"]) >= 1_000_000
    )


@cache
def _small_gemm_kernel(k_size: int, n_size: int, positions: int):
    source = f"""
        constexpr uint K = {k_size};
        constexpr uint N = {n_size};
        constexpr uint P = {positions};
        constexpr uint HEADER_BYTES = {BLOCK_HEADER_BYTES};
        constexpr uint BLOCK_VALUES = {BLOCK_VALUES};
        constexpr uint CODE_BYTES = {CODE_BYTES_PER_BLOCK};
        constexpr uint PATCH_BYTES = {PATCH_BYTES};
        constexpr uint ROWS_PER_SIMD = 4;
        constexpr uint ROWS_PER_TG = 8;
        constexpr uint K_PER_LANE = 16;
        constexpr uint K_BLOCK = 512;

        uint tile = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint row0 = tile * ROWS_PER_TG + simd_gid * ROWS_PER_SIMD;
        if (row0 >= N) return;

        uint codes_base = header_stream_bytes;
        float accum[P][ROWS_PER_SIMD] = {{0.0f}};
        for (uint k_block = 0; k_block < K; k_block += K_BLOCK) {{
            uint k0 = k_block + lane * K_PER_LANE;
            for (uint r = 0; r < ROWS_PER_SIMD; ++r) {{
                uint row = row0 + r;
                if (row >= N) continue;
                uint flat_base = row * K + k0;
                uint block =
                    first_block + flat_base / BLOCK_VALUES;
                uint local_base = flat_base & (BLOCK_VALUES - 1);
                uint header = block * HEADER_BYTES;
                uint patch_offset =
                    uint(encoded[header]) |
                    (uint(encoded[header + 1]) << 8) |
                    (uint(encoded[header + 2]) << 16) |
                    (uint(encoded[header + 3]) << 24);
                uint exception_count =
                    uint(encoded[header + 10]) |
                    (uint(encoded[header + 11]) << 8);
                ushort mode = ushort(encoded[header + 12] & 15);
                #pragma unroll
                for (uint i = 0; i < K_PER_LANE; ++i) {{
                    uint local = local_base + i;
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
                    ushort bits =
                        (code & ushort(0x07ff)) |
                        ((code & ushort(0x0800)) << 4) |
                        (mode << 11);
                    for (
                        uint patch = 0;
                        patch < exception_count;
                        ++patch
                    ) {{
                        uint at = patch_offset + patch * PATCH_BYTES;
                        if (uint(encoded[at]) == local) {{
                            bits =
                                ushort(encoded[at + 1]) |
                                (ushort(encoded[at + 2]) << 8);
                            break;
                        }}
                    }}
                    float weight = as_type<float>(uint(bits) << 16);
                    #pragma unroll
                    for (uint position = 0; position < P; ++position) {{
                        accum[position][r] +=
                            float(x[position * K + k0 + i]) * weight;
                    }}
                }}
            }}
        }}
        for (uint position = 0; position < P; ++position) {{
            for (uint r = 0; r < ROWS_PER_SIMD; ++r) {{
                float value = simd_sum(accum[position][r]);
                uint row = row0 + r;
                if (lane == 0 && row < N) {{
                    out[position * N + row] = T(value);
                }}
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=(
            f"voom_nf12_small_gemm_k{k_size}_n{n_size}_p{positions}"
        ),
        input_names=[
            "x", "encoded", "first_block", "header_stream_bytes"
        ],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=False,
    )


@dataclass
class NF12MappedLayer:
    """Main-consumer-thread lazy mapping shared by one layer's tensors.

    MLX lazy arrays capture the creating thread's default CPU stream. Weight
    fetches run on a prefetch worker, while the fused linear is evaluated on
    the request thread; carrying a worker-created lazy mmap across that boundary
    raises ``There is no Stream(cpu, ...) in current thread``. Carry the path
    across the boundary instead and create the immutable mapping on first
    consumer use.
    """

    path: str
    _encoded: mx.array | None = None

    def array(self) -> mx.array:
        if self._encoded is None:
            loaded = mx.load(str(Path(self.path)))
            encoded = loaded.get("encoded")
            if encoded is None:
                raise ValueError(
                    f"NF12 file {self.path} has no encoded tensor"
                )
            self._encoded = encoded
        return self._encoded


@dataclass
class NF12Tensor:
    encoded: mx.array | NF12MappedLayer
    entry: dict
    spec: dict

    def _encoded_array(self) -> mx.array:
        if isinstance(self.encoded, NF12MappedLayer):
            return self.encoded.array()
        return self.encoded

    @property
    def nbytes(self) -> int:
        return int(self.spec["encoded_bytes"])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.spec["shape"])

    @property
    def dtype(self):
        return mx.bfloat16

    def materialize(self) -> mx.array:
        """Reconstruct this tensor exactly for a non-linear-style consumer.

        Most eligible operands stay compressed through :meth:`matmul`.
        Weight-absorption transforms occasionally need to reshape and slice a
        projection itself (for example MLA's per-head ``kv_b`` factors). Decode
        only this tensor's block span in that case, rather than disabling the
        compact representation for the complete layer.
        """
        from .bf16_nf12_sidecar import decode_names

        return decode_names(
            self._encoded_array(), self.entry, [self.spec["name"]]
        )[self.spec["name"]]

    def reshape(self, *shape: int | tuple[int, ...]) -> mx.array:
        """Match the small part of the dense-array protocol MLA requires."""
        return self.materialize().reshape(*shape)

    def matmul(self, x: mx.array) -> mx.array:
        leading = tuple(int(value) for value in x.shape[:-1])
        positions = 1
        for value in leading:
            positions *= value
        n_size, k_size = self.shape
        encoded = self._encoded_array()
        if (
            positions <= 8
            and x.shape[-1] == k_size
            and direct_linear_eligible(self.spec)
        ):
            flat = x.reshape(positions, k_size)
            output = _small_gemm_kernel(
                k_size, n_size, positions
            )(
                inputs=[
                    flat,
                    encoded,
                    mx.array(
                        int(self.spec["block_start"]),
                        dtype=mx.uint32,
                    ),
                    mx.array(
                        int(self.entry["header_bytes"]),
                        dtype=mx.uint32,
                    ),
                ],
                template=[("T", x.dtype)],
                grid=((n_size + 7) // 8 * 64, 1, 1),
                threadgroup=(64, 1, 1),
                output_shapes=[(positions, n_size)],
                output_dtypes=[x.dtype],
            )[0]
            return output.reshape((*leading, n_size))

        from .bf16_nf12_sidecar import decode_layer

        dense = decode_layer(encoded, self.entry)[self.spec["name"]]
        return x @ dense.T
