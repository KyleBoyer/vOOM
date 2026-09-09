"""Experiment-only raw MXFP4 row ranges; no MLX and no serving integration."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct


@dataclass(frozen=True)
class Extent:
    name: str
    dtype: str
    rows: int
    columns: int
    offset: int
    row_bytes: int

    @property
    def nbytes(self):
        return self.rows * self.row_bytes

    def row_range(self, start, stop):
        if (type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= self.rows):
            raise ValueError('invalid head row interval')
        return self.offset + start * self.row_bytes, (stop - start) * self.row_bytes


def layout(header, data_start, file_bytes):
    if (not isinstance(header, dict) or type(data_start) is not int
            or type(file_bytes) is not int or not 8 <= data_start < file_bytes):
        raise ValueError('invalid head header bounds')
    if 'lm_head.biases' in header:
        raise ValueError('MXFP4 row experiment requires a bias-free head')
    result = []
    for name, dtype, itemsize in (('lm_head.weight', 'U32', 4),
                                  ('lm_head.scales', 'U8', 1)):
        meta = header.get(name)
        if not isinstance(meta, dict) or meta.get('dtype') != dtype:
            raise ValueError('unexpected MXFP4 head physical dtype')
        shape, offsets = meta.get('shape'), meta.get('data_offsets')
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(x) is not int or x <= 0 for x in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(x) is not int for x in offsets)):
            raise ValueError('invalid MXFP4 head shape or offsets')
        start, stop = offsets
        rows, columns = shape
        if (start < 0 or stop - start != rows * columns * itemsize
                or data_start + stop > file_bytes
                or (data_start + start) % itemsize):
            raise ValueError('invalid MXFP4 head byte extent')
        result.append(Extent(name, dtype, rows, columns,
            data_start + start, columns * itemsize))
    weight, scales = result
    if (weight.rows != scales.rows or weight.columns % 4
            or weight.columns // 4 != scales.columns):
        raise ValueError('MXFP4 head requires aligned hidden/32 scale groups')
    if max(weight.offset, scales.offset) < min(
            weight.offset + weight.nbytes, scales.offset + scales.nbytes):
        raise ValueError('overlapping head component extents')
    return weight, scales


def pread_exact(fd, size, offset):
    if type(size) is not int or type(offset) is not int or size <= 0 or offset < 0:
        raise ValueError('invalid read interval')
    first = os.pread(fd, size, offset)
    if len(first) == size:
        return first
    pieces = [first]
    done = len(first)
    while done < size:
        part = os.pread(fd, size - done, offset + done)
        if not part:
            raise EOFError('short head component read')
        pieces.append(part)
        done += len(part)
    return b''.join(pieces)


class HeadRows:
    """One read-only descriptor, strict native MXFP4 layout, bounded header."""
    def __init__(self, directory):
        directory = Path(directory).resolve()
        config = json.loads((directory/'config.json').read_text())
        expected = dict(bits=4, group_size=32, mode='mxfp4')
        if config.get('quantization') != expected or config.get('quantization_config') != expected:
            raise ValueError('row experiment requires explicit native MXFP4 configuration')
        mapping = json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
        shard = mapping.get('lm_head.weight')
        if (not isinstance(shard, str) or not shard or Path(shard).name != shard
                or shard != mapping.get('lm_head.scales')
                or 'lm_head.biases' in mapping):
            raise ValueError('row experiment requires a same-shard bias-free head')
        self.path = directory/shard
        self.fd = os.open(self.path, os.O_RDONLY)
        try:
            self.identity = self.stat_identity()
            size = struct.unpack('<Q', pread_exact(self.fd, 8, 0))[0]
            if not 0 < size <= 16 * 1024**2 or size + 8 >= self.identity[2]:
                raise ValueError('invalid or oversized safetensors header')
            header = json.loads(pread_exact(self.fd, size, 8))
            self.extents = layout(header, size + 8, self.identity[2])
            self.vocab = self.extents[0].rows
            self.hidden = self.extents[0].columns * 8
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def stat_identity(self):
        stat = os.fstat(self.fd)
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def check_unchanged(self):
        if self.stat_identity() != self.identity:
            raise ValueError('head source changed during row experiment')

    def read_component(self, index, start, stop):
        if type(index) is not int or index not in (0, 1):
            raise ValueError('invalid head component')
        self.check_unchanged()
        offset, size = self.extents[index].row_range(start, stop)
        return pread_exact(self.fd, size, offset)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
