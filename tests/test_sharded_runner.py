"""Pure watchdog tests for the low-memory pytest process sharder."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


_PATH = Path(__file__).with_name("run_pytest_sharded.py")
_SPEC = importlib.util.spec_from_file_location("run_pytest_sharded", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


class FakeProcess:
    pid = 1234

    def __init__(self, returncode):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def test_completed_shard_returns_without_refusal(monkeypatch, tmp_path):
    proc = FakeProcess(0)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(
        runner.psutil, "swap_memory", lambda: SimpleNamespace(used=100))

    result = runner._run_shard(
        ["pytest"], tmp_path, {}, int(4e9), int(16e6))

    assert result == (0, None)


def test_shard_tree_is_stopped_below_available_floor(monkeypatch, tmp_path):
    proc = FakeProcess(None)
    stopped = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(
        runner.psutil, "virtual_memory",
        lambda: SimpleNamespace(available=int(3.9e9)))
    monkeypatch.setattr(
        runner.psutil, "swap_memory", lambda: SimpleNamespace(used=100))
    monkeypatch.setattr(runner, "_stop_process_group", stopped.append)

    returncode, refusal = runner._run_shard(
        ["pytest"], tmp_path, {}, int(4e9), int(16e6))

    assert returncode == 2
    assert "available memory" in refusal
    assert stopped == [proc]


def test_shard_tree_is_stopped_on_net_swap_growth(monkeypatch, tmp_path):
    proc = FakeProcess(None)
    stopped = []
    swap_used = iter((100, int(17e6) + 100))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(
        runner.psutil, "virtual_memory",
        lambda: SimpleNamespace(available=int(8e9)))
    monkeypatch.setattr(
        runner.psutil, "swap_memory",
        lambda: SimpleNamespace(used=next(swap_used)))
    monkeypatch.setattr(runner, "_stop_process_group", stopped.append)

    returncode, refusal = runner._run_shard(
        ["pytest"], tmp_path, {}, int(4e9), int(16e6))

    assert returncode == 2
    assert "swap occupancy" in refusal
    assert stopped == [proc]
