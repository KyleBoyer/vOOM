"""Exact ephemeral rollback storage; no model weights required."""
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest

from runtime.kda_state import KDAStateCache, KDAFactorStep, KDAFactorWindow
from runtime import qwen_mtp_base_spill as spill


def payload(value):
    return None if value is None else KDAStateCache._array_payload(value)


def raw(cache):
    return [(payload(cache.state(i)), None if cache.conv_history(i) is None
             else tuple(payload(v) for v in cache.conv_history(i)))
            for i in range(len(cache._state))]


def source():
    cache = KDAStateCache(3)
    for layer in (0, 2):
        cache.set_state(layer, mx.arange(32, dtype=mx.float32).reshape(1, 2, 4, 4) / 16)
        cache.set_conv_history(layer, (mx.array([0., -0., 1.], dtype=mx.bfloat16),
                                      None, mx.array([2., -2.], dtype=mx.float16)))
    return cache


@pytest.mark.parametrize('prefix', range(6))
def test_every_scalar_prefix_raw_bytes(tmp_path, prefix):
    cache = source()
    original = raw(cache)
    steps = []
    for layer in range(3):
        steps.append([] if layer == 1 else [KDAFactorStep(
            gate=mx.full((1, 2), -.125), key=mx.full((1, 2, 4), .25),
            value=mx.full((1, 2, 4), position / 7), beta=mx.full((1, 2), .5),
            conv_history=(mx.full((1, 3, 8), position, dtype=mx.bfloat16),))
            for position in range(5)])
    window = KDAFactorWindow(steps, 5)
    stats, governor = {}, Mock()
    disk = spill.DiskFactorBase(cache, governor, stats, tmp_path)
    directory = disk._temporary.name
    assert disk.nbytes() == 0 and spill.base_bytes(disk) == cache.nbytes()
    assert [spill.has_state(disk, i) for i in range(3)] == [True, False, True]
    assert raw(window.commit_prefix(disk, prefix)) == raw(window.commit_prefix(cache, prefix))
    assert raw(cache) == original and disk.nbytes() == 0
    assert stats['bytes_written'] == stats['bytes_read'] == cache.nbytes()
    assert stats['reloads'] == 1
    governor.reserve.assert_called_once_with(cache.nbytes() + 6 * 128,
        margin=0, reason='qwen-mtp-base-reload')
    disk.close()
    disk.close()
    assert not spill.Path(directory).exists() and stats['closed'] == 1


@pytest.mark.parametrize('failure', ['success', 'error'])
def test_generation_owner_closes_without_reload(tmp_path, monkeypatch, failure):
    monkeypatch.setenv(spill.FLAG, '1')
    monkeypatch.setattr(spill, 'ROOT', tmp_path)
    target = SimpleNamespace(governor=Mock())
    @spill.managed_bases
    def generate():
        disk = spill.create_base(source(), target)
        if failure == 'error':
            raise RuntimeError('partially advanced endpoint')
        spill.close_base(disk)
    if failure == 'error':
        with pytest.raises(RuntimeError, match='partially advanced'):
            generate()
    else:
        generate()
    assert list(tmp_path.iterdir()) == []
    assert target._qwen_mtp_base_spill_stats['closed'] == 1
    assert target._qwen_mtp_base_spill_stats.get('reloads', 0) == 0


@pytest.mark.parametrize('damage', ['truncate', 'corrupt'])
def test_corruption_fails_closed(tmp_path, damage):
    cache = source()
    before = raw(cache)
    disk = spill.DiskFactorBase(cache, Mock(), {}, tmp_path)
    path = disk._records[0]['arrays']['state'][0]
    with path.open('r+b') as output:
        if damage == 'truncate':
            output.truncate(1)
        else:
            output.write(b'xxxx')
    with pytest.raises(IOError, match='snapshot'):
        disk.fork()
    assert raw(cache) == before
    disk.close()


def test_admission_precedes_reload(tmp_path, monkeypatch):
    governor = Mock()
    governor.reserve.side_effect = MemoryError('ordinary reserve')
    disk = spill.DiskFactorBase(source(), governor, {}, tmp_path)
    convert = Mock(side_effect=AssertionError('must not allocate'))
    monkeypatch.setattr(KDAStateCache, '_array_from_payload', convert)
    with pytest.raises(MemoryError, match='ordinary reserve'):
        disk.fork()
    convert.assert_not_called()
    disk.close()


def test_default_off_and_fail_closed(monkeypatch):
    monkeypatch.delenv(spill.FLAG, raising=False)
    cache = source()
    base = spill.create_base(cache, SimpleNamespace())
    assert base.state(0) is cache.state(0)
    monkeypatch.setenv(spill.FLAG, 'yes')
    with pytest.raises(ValueError, match='0 or 1'):
        spill.create_base(cache, SimpleNamespace())
    monkeypatch.setenv(spill.FLAG, '1')
    with pytest.raises(RuntimeError, match='generation owner'):
        spill.create_base(cache, SimpleNamespace())


def test_disk_reserve_before_files(tmp_path, monkeypatch):
    monkeypatch.setattr(spill.psutil, 'disk_usage', lambda _: SimpleNamespace(free=1))
    with pytest.raises(MemoryError, match='free disk'):
        spill.DiskFactorBase(source(), Mock(), {}, tmp_path)
    assert list(tmp_path.iterdir()) == []
