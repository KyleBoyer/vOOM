"""CPU-only fixture bounds/import/positive-control tests; no MLX import."""

import importlib.util
from pathlib import Path
import sys
import weakref

import pytest

PATH = Path(__file__).parent / "fixtures" / "host_buffer_lifetime_probe.py"
spec = importlib.util.spec_from_file_location("host_buffer_lifetime_probe", PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("mib", [1, 16, 32])
def test_exact_bounded_payload_shape(mib):
    rows, columns = probe.buffer_shape(mib)
    assert rows * columns * 2 == mib * 1024 * 1024
    assert columns == 65536


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 33, 1.0, "16", None])
def test_invalid_size_rejected(invalid):
    with pytest.raises(ValueError):
        probe.buffer_shape(invalid)


def test_weakref_standin_observes_view_retention_and_release():
    owner = probe.TrackedBuffer(1024)
    ref = weakref.ref(owner)
    view = memoryview(owner)[-16:]
    del owner
    assert ref() is not None
    del view
    assert ref() is None
