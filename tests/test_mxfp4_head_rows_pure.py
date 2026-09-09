"""No MLX: exact MXFP4 row arithmetic, malformed-layout and short-read gates."""

import json
import os
import struct
from types import SimpleNamespace

import pytest

from runtime import mxfp4_head_rows as rows


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


def test_bias_and_truncated_file_rejected():
    for value, start, size in [(dict(header(), **{'lm_head.biases': {}}), 128, 230),
                              (header(), 128, 229)]:
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


def make_source(tmp_path, data_start_mod4=0):
    q = dict(bits=4, group_size=32, mode='mxfp4')
    (tmp_path/'config.json').write_text(json.dumps(dict(quantization=q, quantization_config=q)))
    mapping = {key:'head.safetensors' for key in header()}
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map=mapping)))
    encoded = json.dumps(header()).encode()
    encoded += b' ' * ((data_start_mod4 - 8 - len(encoded)) % 4)
    payload = bytes(range(102))
    (tmp_path/'head.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
    return payload


@pytest.mark.parametrize('data_start_mod4', [0, 1, 2, 3])
def test_actual_row_reader_consumes_only_declared_ranges_and_closes(tmp_path, data_start_mod4):
    payload = make_source(tmp_path, data_start_mod4)
    source = rows.HeadRows(tmp_path)
    fd = source.fd
    assert source.extents[0].offset % 4 == data_start_mod4
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


def native_store():
    return SimpleNamespace(vpack2=None, packed={}, gguf=None, fast_dirs=[],
        _raw_fast_tier_manifest={}, k3_scale_sidecar=None, bf16_nf12_sidecar=None,
        _ct_int4_aux={}, _ct_mxfp4_aux={}, _glm53_fp8_aux={}, _dsv4_aux={},
        _qwen4_fused_expert_slices={}, _quant_aux={'lm_head.weight':
            SimpleNamespace(bits=4, group_size=32, mode='mxfp4',
                scales='lm_head.scales', biases=None)})


def test_native_reference_budget_does_not_count_a_nonexistent_whole_host_copy():
    from tests.fixtures.huihui_mxfp4_head_rows_gate import native_reference_reservation_bytes
    assert native_reference_reservation_bytes(native_store(), 675430400, '0.32.0') == 675430400


@pytest.mark.parametrize('name', ['vpack2', 'packed', 'gguf', 'fast_dirs',
    '_raw_fast_tier_manifest', 'k3_scale_sidecar', 'bf16_nf12_sidecar',
    '_ct_int4_aux', '_ct_mxfp4_aux', '_glm53_fp8_aux', '_dsv4_aux',
    '_qwen4_fused_expert_slices'])
def test_other_loaders_do_not_inherit_native_reference_budget(name):
    from tests.fixtures.huihui_mxfp4_head_rows_gate import native_reference_reservation_bytes
    store = native_store()
    setattr(store, name, True)
    with pytest.raises(ValueError, match='native raw'):
        native_reference_reservation_bytes(store, 675430400, '0.32.0')
    delattr(store, name)
    with pytest.raises(ValueError, match='native raw'):
        native_reference_reservation_bytes(store, 675430400, '0.32.0')


@pytest.mark.parametrize('change', ['version', 'size', 'bias', 'bits', 'missing'])
def test_reference_budget_is_version_geometry_and_representation_pinned(change):
    from tests.fixtures.huihui_mxfp4_head_rows_gate import native_reference_reservation_bytes
    store, size, version = native_store(), 675430400, '0.32.0'
    if change == 'version':
        version = 'unknown'
    elif change == 'size':
        size -= 1
    elif change == 'bias':
        store._quant_aux['lm_head.weight'].biases = 'lm_head.biases'
    elif change == 'bits':
        store._quant_aux['lm_head.weight'].bits = 8
    else:
        store._quant_aux.clear()
    with pytest.raises(ValueError):
        native_reference_reservation_bytes(store, size, version)
