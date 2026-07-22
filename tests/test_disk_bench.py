import os
from unittest.mock import patch

from runtime.disk_bench import (measure_disk_profile,
                                measure_scattered_mb_per_s)


def test_scattered_profile_reports_uncached_application(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(os.urandom(256 * 1024))
    with patch("runtime.disk_bench._set_uncached", return_value=True) as apply:
        curve, applied = measure_scattered_mb_per_s(
            path, chunk_sizes=(4096, 16384), target_bytes=64 * 1024,
            min_file_bytes=1, uncached=True, seed=7)
    assert applied
    assert set(curve) == {4096, 16384}
    assert all(value > 0 for value in curve.values())
    apply.assert_called_once()


def test_disk_profile_keeps_uncached_truth_in_result(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(os.urandom(256 * 1024))
    with patch("runtime.disk_bench._set_uncached", return_value=False):
        profile = measure_disk_profile(
            path, sample_bytes=64 * 1024,
            scattered_target_bytes=64 * 1024, min_file_bytes=1,
            uncached=True)
    assert profile.uncached_requested
    assert not profile.uncached_applied
    assert profile.sequential_mb_per_s > 0
    assert profile.scattered_mb_per_s
