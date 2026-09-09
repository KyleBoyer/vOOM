"""No MLX: exact MXFP4 row arithmetic, malformed-layout and short-read gates."""

import json
import os
import struct

import pytest

from tests.fixtures import mxfp4_head_rows as rows


def header():
    return {'lm_head.weight': dict(dtype='U32', shape=[3, 8], data_offsets=[0, 96]),
            'lm_head.scales': dict(dtype='U8', shape=[3, 2], data_offsets=[96, 102])}


def test_row_ranges_are_exact_and_preserve_partial_final_block():
    weight, scales = rows.layout(header(), 128, 230)
    assert weight.row_range(1, 3) == (160, 64)
    assert scales.row_range(2, 3) == (228, 2)
    assert weight.nbytes + scales.nbytes == 102
    assert weight.columns * 8 == scales.columns * 32 == 64


@pytest.mark.parametrize('start,stop', [(True, 2), (0, True), (-1, 1), (0, 4), (2, 2), (2, 1)])
def test_invalid_row_interval_rejected(start, stop):
    with pytest.raises(ValueError):
        rows.layout(header(), 128, 230)[0].row_range(start, stop)


@pytest.mark.parametrize('component,key,value', [
    ('lm_head.weight', 'dtype', 'BF16'), ('lm_head.scales', 'dtype', 'F32'),
    ('lm_head.weight', 'shape', [True, 8]), ('lm_head.weight', 'shape', [3, 0]),
    ('lm_head.scales', 'shape', [3, 1]), ('lm_head.scales', 'data_offsets', [95, 101]),
    ('lm_head.weight', 'data_offsets', [-1, 95]),
    ('lm_head.weight', 'data_offsets', [0, 95]),
    ('lm_head.weight', 'data_offsets', [True, 96]),
    ('lm_head.weight', 'data_offsets', [4, 100])])
def test_bad_physical_layout_rejected(component, key, value):
    value_header = header()
    value_header[component][key] = value
    with pytest.raises(ValueError):
        rows.layout(value_header, 128, 230)


def test_overlap_bias_unaligned_and_truncated_file_rejected():
    for value, start, size in [(dict(header(), **{'lm_head.biases': {}}), 128, 230),
                              (header(), 129, 231), (header(), 128, 229)]:
        with pytest.raises(ValueError):
            rows.layout(value, start, size)


def test_exact_pread_retries_short_reads_and_rejects_eof(monkeypatch):
    chunks = iter([b'ab', b'c', b'd'])
    seen = []
    def pread(fd, size, offset):
        seen.append((size, offset))
        return next(chunks)
    monkeypatch.setattr(rows.os, 'pread', pread)
    assert rows.pread_exact(7, 4, 10) == b'abcd'
    assert seen == [(4, 10), (2, 12), (1, 13)]
    monkeypatch.setattr(rows.os, 'pread', lambda *a: b'')
    with pytest.raises(EOFError):
        rows.pread_exact(7, 4, 10)


def make_source(tmp_path):
    q = dict(bits=4, group_size=32, mode='mxfp4')
    (tmp_path/'config.json').write_text(json.dumps(dict(quantization=q, quantization_config=q)))
    mapping = {key:'head.safetensors' for key in header()}
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=mapping)))
    encoded = json.dumps(header()).encode()
    encoded += b' ' * ((-len(encoded)) % 8)
    payload = bytes(range(102))
    (tmp_path/'head.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
    return payload


def test_actual_row_reader_consumes_only_declared_ranges_and_closes(tmp_path):
    payload = make_source(tmp_path)
    source = rows.HeadRows(tmp_path)
    fd = source.fd
    assert source.read_component(0, 1, 3) == payload[32:96]
    assert source.read_component(1, 2, 3) == payload[100:102]
    source.check_unchanged()
    source.close()
    source.close()
    with pytest.raises(OSError):
        os.fstat(fd)


def test_mutation_is_not_silently_read(tmp_path):
    make_source(tmp_path)
    source = rows.HeadRows(tmp_path)
    try:
        with source.path.open('ab') as handle:
            handle.write(b'changed')
        with pytest.raises(ValueError, match='changed'):
            source.read_component(0, 0, 1)
    finally:
        source.close()


@pytest.mark.parametrize('change', ['path', 'split', 'bias', 'mode', 'large_header'])
def test_open_fails_closed_on_unsupported_or_oversized_source(tmp_path, change):
    make_source(tmp_path)
    index_path = tmp_path/'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    if change == 'path':
        index['weight_map']['lm_head.weight'] = '../head.safetensors'
    elif change == 'split':
        index['weight_map']['lm_head.scales'] = 'other.safetensors'
    elif change == 'bias':
        index['weight_map']['lm_head.biases'] = 'head.safetensors'
    elif change == 'mode':
        (tmp_path/'config.json').write_text('{}')
    else:
        (tmp_path/'head.safetensors').write_bytes(struct.pack('<Q', 2**40))
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        rows.HeadRows(tmp_path)


@pytest.mark.parametrize('change', ['failed', 'short', 'stale', 'transcoder', 'disk'])
def test_numeric_gate_requires_fresh_full_preflight_before_mlx_or_weights(tmp_path, change):
    from tests.fixtures import huihui_mxfp4_head_rows_gate as gate
    pre = dict(passed=True, sample_seconds=30, known_transcoders={'passed': True},
        end=dict(monotonic_s=gate.time.monotonic(), root_free_bytes=20_000_000_000))
    if change == 'failed':
        pre['passed'] = False
    elif change == 'short':
        pre['sample_seconds'] = 29
    elif change == 'stale':
        pre['end']['monotonic_s'] -= 121
    elif change == 'transcoder':
        pre['known_transcoders']['passed'] = False
    else:
        pre['end']['root_free_bytes'] = 9_999_999_999
    path = tmp_path/'preflight.json'
    path.write_text(json.dumps(pre))
    with pytest.raises(AssertionError):
        gate.run(path, tmp_path/'result.json')
    assert not (tmp_path/'result.json').exists()
