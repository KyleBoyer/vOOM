"""Pure regression gates for F42's pre-allocation reservation.

The real module normally imports MLX, so this test loads it with a tiny fake
``mlx.core`` module. No Metal framework or model code is imported.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class FakeMX(types.ModuleType):
    def __init__(self, active: int):
        super().__init__("mlx.core")
        self.active = active
        self.clears = 0
        self.info = {}

    def get_active_memory(self):
        return self.active

    def clear_cache(self):
        self.clears += 1

    def device_info(self):
        return self.info


class FakeCache:
    def __init__(self, mx, max_bytes: int, floor_after_evict: int | None = None):
        self.mx = mx
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self.floor_after_evict = floor_after_evict
        self.total_bytes = 0

    def _evict_locked(self):
        if self.floor_after_evict is not None:
            self.mx.active = min(self.mx.active, self.floor_after_evict)


class FakePrefetcher:
    paused = False


def load_pressure(active: int):
    fake_core = FakeMX(active)
    fake_pkg = types.ModuleType("mlx")
    fake_pkg.core = fake_core
    old_mlx = sys.modules.get("mlx")
    old_core = sys.modules.get("mlx.core")
    sys.modules["mlx"] = fake_pkg
    sys.modules["mlx.core"] = fake_core
    try:
        name = f"_vmodel_pressure_test_{active}_{id(fake_core)}"
        spec = importlib.util.spec_from_file_location(name, ROOT / "runtime" / "pressure.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if old_mlx is None:
            sys.modules.pop("mlx", None)
        else:
            sys.modules["mlx"] = old_mlx
        if old_core is None:
            sys.modules.pop("mlx.core", None)
        else:
            sys.modules["mlx.core"] = old_core
    return module, fake_core


def make_governor(module, fake_mx, *, cache_max: int, floor: int,
                  active_after_evict: int | None = None):
    gov = module.MemoryGovernor.__new__(module.MemoryGovernor)
    gov.cache = FakeCache(fake_mx, cache_max, active_after_evict)
    gov.prefetcher = FakePrefetcher()
    gov.floor = floor
    gov.critical = int(1.2e9)
    gov.metal_limit = int(10e9)
    gov.reservations = 0
    gov.reservation_failures = 0
    gov.configured_max = cache_max
    gov.paused_prefetch = False
    return gov


def test_safe_projection_is_noop():
    module, mx = load_pressure(int(5e9))
    gov = make_governor(module, mx, cache_max=int(4e9), floor=int(1.5e9))
    gov.reserve(int(1e9), margin=int(0.4e9))
    assert gov.cache.max_bytes == int(4e9)
    assert gov.reservations == 0
    assert gov.reservation_failures == 0
    assert not gov.prefetcher.paused


def test_reservation_reclaims_then_allows():
    module, mx = load_pressure(int(9.8e9))
    gov = make_governor(
        module, mx, cache_max=int(5e9), floor=int(1.5e9),
        active_after_evict=int(9.0e9),
    )
    gov.reserve(int(0.3e9), margin=int(0.4e9))
    assert gov.cache.max_bytes < int(5e9)
    assert gov.reservations == 1
    assert gov.reservation_failures == 0
    assert gov.prefetcher.paused


def test_unreclaimable_projection_fails_before_allocation():
    module, mx = load_pressure(int(9.9e9))
    gov = make_governor(module, mx, cache_max=int(1.5e9), floor=int(1.5e9))
    try:
        gov.reserve(int(1e9), margin=int(0.4e9))
    except MemoryError as exc:
        assert "refused before allocation" in str(exc)
        assert "projected=11.30GB" in str(exc)
    else:
        raise AssertionError("unsafe reservation was allowed to continue")
    assert gov.reservation_failures == 1
    assert gov.prefetcher.paused


def test_live_ceiling_uses_device_limit_and_sampled_system_headroom():
    module, mx = load_pressure(int(8e9))
    gov = make_governor(module, mx, cache_max=int(5e9), floor=int(1.5e9))
    gov.metal_limit = int(40e9)

    assert gov._metal_ceiling(int(8e9), int(33e9)) == int(39.8e9)
    assert gov._metal_ceiling(int(8e9), int(2e9)) == int(8.8e9)


def test_device_limit_uses_mlx_metadata_instead_of_a_fixed_host_cap():
    module, mx = load_pressure(0)
    mx.info = {
        "max_recommended_working_set_size": int(55e9),
        "memory_size": int(64e9),
    }
    assert module._device_recommended_limit() == int(55e9)

    mx.info = {"memory_size": int(64e9)}
    assert module._device_recommended_limit() == int(64e9)


def test_cache_target_is_fitted_to_sampled_live_headroom():
    module, mx = load_pressure(int(5e9))
    gov = make_governor(module, mx, cache_max=int(9e9), floor=int(1.5e9))
    gov.cache.total_bytes = int(0.5e9)
    original_virtual_memory = module.psutil.virtual_memory
    try:
        module.psutil.virtual_memory = lambda: types.SimpleNamespace(
            available=int(4e9))
        fitted = gov.fit_cache_to_live_headroom(margin=int(0.4e9))
    finally:
        # ``module.psutil`` is the process-global imported package. Restore the
        # patched callable so later memory-planner tests still see ``total``.
        module.psutil.virtual_memory = original_virtual_memory

    # ceiling = active 5 + (available 4 - critical 1.2) = 7.8;
    # additional = 7.8 - active 5 - margin .4 = 2.4; plus .5 resident.
    assert fitted == int(2.9e9)
    assert gov.reservations == 1
    assert gov.prefetcher.paused


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:
            failures.append((test.__name__, exc))
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"PASS {len(tests) - len(failures)}/{len(tests)}; fake MLX/no Metal")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
