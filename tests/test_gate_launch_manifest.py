"""Launch provenance survives interrupted gate-runner bookkeeping."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures import run_gate


def _args(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "fixture.py").write_text("VALUE = 1\n")
    return ["--result-dir", str(tmp_path / "results"), "--run-id", "launch",
            "--proof-class", "L0", "--expected", "launch provenance",
            "--source-root", str(source), "--cwd", str(tmp_path),
            "--poll-seconds", "0.02"]


def test_manifest_is_durable_before_environment_or_child(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_gate.py", *args, "--", "never-launched"])

    def interrupted():
        raise RuntimeError("simulated interruption before child launch")

    monkeypatch.setattr(run_gate, "package_snapshot", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_gate.main()
    running = json.loads((tmp_path / "results/launch.running.json").read_text())
    assert running["child_pid"] is None
    assert running["state"] == "running"
    assert running["source_manifest"] == run_gate.source_manifest(tmp_path / "source")
    assert not (tmp_path / "results/launch.done.json").exists()


def test_child_observes_same_launch_manifest_as_final_record(tmp_path):
    args = _args(tmp_path)
    child = ("import json; from pathlib import Path; "
             "d=json.loads(Path('results/launch.running.json').read_text()); "
             "print(json.dumps(d['source_manifest']))")
    completed = subprocess.run(
        [sys.executable, str(Path(run_gate.__file__).resolve()), *args,
         "--", sys.executable, "-c", child], capture_output=True, text=True,
        timeout=30)
    assert completed.returncode == 0, completed.stderr
    done = json.loads((tmp_path / "results/launch.done.json").read_text())
    observed = json.loads(Path(done["log_path"]).read_text())
    assert observed == done["source_manifest"] == done["source_manifest_end"]
    assert done["verdict"] == "PASS"
    assert not (tmp_path / "results/launch.running.json").exists()
