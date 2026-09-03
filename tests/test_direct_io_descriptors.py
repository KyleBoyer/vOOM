import os
import threading

from runtime.model_loader import WeightStore


def _bare_store():
    store = object.__new__(WeightStore)
    store._stage_lock = threading.Lock()
    store._raw_fast_tier_executor = None
    store._direct_fd_lock = threading.Lock()
    store._direct_fds = {}
    store._direct_fd_cache_enabled = True
    store._direct_fd_nocache = False
    store.direct_fd_opens = 0
    store.direct_fd_hits = 0
    store.direct_fd_closes = 0
    store.direct_fd_open_ns = 0
    store.direct_fd_nocache_applied = 0
    store.direct_pread_calls = 0
    store.direct_pread_requested_bytes = 0
    store.direct_pread_bytes = 0
    store.direct_pread_ns = 0
    store.direct_pread_short_reads = 0
    return store


def test_direct_descriptor_is_reused_and_closed(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"0123456789")
    store = _bare_store()
    first, size = store._direct_fd(path)
    assert size == 10
    assert store._pread_exact(first, 4, 3) == b"3456"
    second, size = store._direct_fd(path)
    assert (first, size) == (second, 10)
    snapshot = store.direct_io_snapshot()
    assert snapshot["fd_opens"] == 1
    assert snapshot["fd_hits"] == 1
    assert snapshot["fd_cached"] == 1
    assert snapshot["pread_calls"] == 1
    assert snapshot["pread_requested_bytes"] == 4
    assert snapshot["pread_bytes"] == 4
    store.close()
    assert store.direct_io_snapshot()["fd_cached"] == 0
    try:
        os.fstat(first)
    except OSError:
        pass
    else:
        raise AssertionError("WeightStore.close() leaked a descriptor")


def test_direct_read_fails_closed_on_truncation(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"small")
    store = _bare_store()
    descriptor, _size = store._direct_fd(path)
    try:
        try:
            store._pread_exact(descriptor, 20, 0)
        except IOError as error:
            assert "truncated direct read" in str(error)
        else:
            raise AssertionError("truncated direct read was accepted")
    finally:
        store.close()


def test_direct_descriptor_cache_has_a_real_disabled_control(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"0123456789")
    store = _bare_store()
    store._direct_fd_cache_enabled = False
    for _ in range(2):
        with store._direct_reader(path) as (descriptor, size):
            assert size == 10
            assert store._pread_exact(descriptor, 2, 4) == b"45"
    snapshot = store.direct_io_snapshot()
    assert snapshot["fd_cache_enabled"] == 0
    assert snapshot["fd_opens"] == 2
    assert snapshot["fd_hits"] == 0
    assert snapshot["fd_closes"] == 2
    assert snapshot["fd_cached"] == 0
    store.close()
