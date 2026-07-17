#!/usr/bin/env python3
"""Pure subprocess tests for the crash-visible gate wrapper."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.fixtures.run_gate import paired_log_ratio_summary


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "fixtures" / "run_gate.py"


def _invoke(out: Path, run_id: str, child: list[str], *extra: str):
    return subprocess.run(
        [
            sys.executable, str(RUNNER), "--result-dir", str(out),
            "--run-id", run_id, "--proof-class", "E",
            "--expected", "pure runner regression", "--poll-seconds", "0.02",
            *extra, "--", *child,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def _done(out: Path, run_id: str) -> dict:
    path = out / f"{run_id}.done.json"
    assert path.exists()
    assert not (out / f"{run_id}.done.json.tmp").exists()
    assert not (out / f"{run_id}.running.json").exists()
    return json.loads(path.read_text())


def _artifact(result: dict) -> Path:
    artifact = Path(result["artifact_dir"])
    assert artifact.is_dir()
    assert artifact.name.endswith(".artifacts")
    assert json.loads((artifact / "manifest.json").read_text())["run_id"] == result["run_id"]
    return artifact


def _claim_payload(
    *, kind: str = "performance", passed: bool = True, pairs: int = 6,
) -> dict:
    observations = []
    for index in range(pairs):
        order = (["control", "candidate"] if index % 2 == 0
                 else ["candidate", "control"])
        observations.append({
            "pair_id": f"pair-{index}",
            "order": order,
            "control": {
                "metrics": {"total_seconds": 2.0 + index / 10},
                "tokens": [1, 2, 3],
            },
            "candidate": {
                "metrics": {"total_seconds": 1.0 + index / 20},
                "tokens": [1, 2, 3],
            },
        })
    payload = {
        "schema": "voom.gate.observations.v1",
        "claim_kind": kind,
        "status": "measured",
        "seed": 17,
        "fingerprints": {
            "model": {"manifest_sha256": "model-sha"},
            "config": {"sha256": "config-sha"},
        },
        "decision": {
            "passed": passed,
            "rule": "candidate is exact and paired ratio clears frozen margin",
            "minimum_ratio": 1.03,
        },
        "observations": observations,
    }
    if "performance" in kind:
        payload["primary_metric"] = {
            "name": "total_seconds",
            "path": "metrics.total_seconds",
            "higher_is_better": False,
        }
    return payload


def _json_writer(path: Path, value: dict) -> list[str]:
    code = (
        "import pathlib,sys; "
        "pathlib.Path(sys.argv[1]).write_text(sys.argv[2])"
    )
    return [sys.executable, "-c", code, str(path), json.dumps(value)]


def test_backward_compatible_smoke_run_without_result() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        proc = _invoke(out, "pass", [sys.executable, "-c", "print('ok')"])
        assert proc.returncode == 0, proc.stderr
        result = _done(out, "pass")
        assert result["verdict"] == "PASS" and result["exit_code"] == 0
        assert result["run_mode"] == "smoke"
        assert result["measurement_status"] == "not_requested"
        assert result["source_manifest"]["file_count"] > 0
        assert result["log_sha256"] == hashlib.sha256(b"ok\n").hexdigest()
        assert result["log_limit_exceeded"] is False
        artifact = _artifact(result)
        assert (artifact / "stdout.raw").read_bytes() == b"ok\n"
        assert (artifact / "stderr.raw").read_bytes() == b""


def test_nonzero_result() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        proc = _invoke(out, "fail", [sys.executable, "-c", "raise SystemExit(7)"])
        assert proc.returncode == 1
        result = _done(out, "fail")
        assert result["verdict"] == "FAIL" and result["exit_code"] == 7
        _artifact(result)


def test_timeout_records_signal() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        proc = _invoke(
            out, "timeout", [sys.executable, "-c", "import time; time.sleep(5)"],
            "--timeout", "0.08", "--kill-grace", "0.05",
        )
        assert proc.returncode == 1
        result = _done(out, "timeout")
        assert result["verdict"] == "FAIL" and result["timed_out"] is True
        assert result["signal"] in (15, 9)


def test_legacy_child_result_is_fresh_and_ingested_for_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        child_result = out / "child.json"
        payload = {"tokens": [1, 2], "true_peak_metal_bytes": 123}
        proc = _invoke(
            out, "child-result", _json_writer(child_result, payload),
            "--child-result-json", str(child_result),
        )
        assert proc.returncode == 0, proc.stderr
        result = _done(out, "child-result")
        assert result["verdict"] == "PASS"
        assert result["measurement_status"] == "provided_unvalidated"
        assert result["child_result"]["tokens"] == [1, 2]
        assert result["child_result_sha256"]
        assert result["child_result_error"] is None
        artifact = _artifact(result)
        assert json.loads((artifact / "structured-result.raw.json").read_text()) == payload


def test_measured_claim_records_observations_summary_lock_and_fingerprints() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "out"
        out.mkdir()
        child_result = root / "claim.json"
        payload = _claim_payload()
        model_config = root / "config.json"
        model_config.write_text('{"model_type":"fixture"}\n')
        lock = root / "benchmark.lock"
        lock.mkdir()
        (lock / "owner").write_text(str(os.getpid()))
        proc = _invoke(
            out, "claim", _json_writer(child_result, payload),
            "--claim-kind", "performance",
            "--child-result-json", str(child_result),
            "--fingerprint", f"model_config={model_config}",
            "--lock-path", str(lock), "--require-lock",
        )
        assert proc.returncode == 0, proc.stderr
        result = _done(out, "claim")
        assert result["verdict"] == "PASS"
        assert result["measurement_status"] == "measured"
        assert result["lock"]["owner_pid"] == os.getpid()
        assert result["lock"]["owner_alive"] is True
        assert result["lock_changed_before_child"] is False
        assert result["lock_changed_during_run"] is False
        assert result["lock_error"] is None
        assert result["input_fingerprints"]["model_config"]["sha256"] == hashlib.sha256(
            model_config.read_bytes()).hexdigest()
        details = result["claim_details"]
        assert details["metadata"]["seed"] == 17
        assert details["metadata"]["observation_count"] == 6
        assert details["metadata"]["order"][0] == ["control", "candidate"]
        summary = details["paired_summary"]
        assert summary["pairs"] == 6 and summary["wins"] == 6
        assert summary["ratio_median"] == 2.0
        assert summary["confidence_interval"]["ratio_lower"] == 2.0
        assert summary["decision"]["lower_bound_passed"] is True
        assert result["child_result"]["observations"] == payload["observations"]
        artifact = _artifact(result)
        assert (artifact / "structured-result.raw.json").read_text() == json.dumps(payload)


def test_declared_claim_without_result_fails_as_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        proc = _invoke(
            out, "claim-missing", [sys.executable, "-c", "raise SystemExit(0)"],
            "--claim-kind", "performance",
        )
        assert proc.returncode == 1
        result = _done(out, "claim-missing")
        assert result["verdict"] == "FAIL"
        assert result["measurement_status"] == "missing"
        assert "requires --child-result-json" in result["child_result_error"]


def test_missing_requested_child_result_turns_zero_exit_into_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        missing = out / "missing.json"
        proc = _invoke(
            out, "missing-child", [sys.executable, "-c", "raise SystemExit(0)"],
            "--child-result-json", str(missing),
        )
        assert proc.returncode == 1
        result = _done(out, "missing-child")
        assert result["verdict"] == "FAIL" and result["exit_code"] == 0
        assert result["measurement_status"] == "missing"
        assert result["child_result_error"]


def test_malformed_claim_result_fails_and_preserves_raw_bytes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "out"
        child_result = root / "malformed.json"
        raw = b'{"schema": nope\n'
        code = (
            "import pathlib,sys; "
            "pathlib.Path(sys.argv[1]).write_bytes(bytes.fromhex(sys.argv[2]))"
        )
        proc = _invoke(
            out, "malformed",
            [sys.executable, "-c", code, str(child_result), raw.hex()],
            "--claim-kind", "quality",
            "--child-result-json", str(child_result),
        )
        assert proc.returncode == 1
        result = _done(out, "malformed")
        assert result["verdict"] == "FAIL"
        assert result["measurement_status"] == "malformed"
        assert "JSONDecodeError" in result["child_result_error"]
        artifact = _artifact(result)
        assert (artifact / "structured-result.raw.json").read_bytes() == raw


def test_explicit_unmeasured_claim_is_distinct_and_nonzero() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "out"
        child_result = root / "unmeasured.json"
        payload = {
            "schema": "voom.gate.observations.v1",
            "claim_kind": "quality",
            "status": "unmeasured",
            "reason": "reference corpus unavailable",
        }
        proc = _invoke(
            out, "unmeasured", _json_writer(child_result, payload),
            "--claim-kind", "quality",
            "--child-result-json", str(child_result),
        )
        assert proc.returncode == 1
        result = _done(out, "unmeasured")
        assert result["verdict"] == "UNMEASURED"
        assert result["measurement_status"] == "unmeasured"
        assert result["child_result_error"] is None
        assert result["claim_details"]["reason"] == "reference corpus unavailable"


def test_measured_quality_claim_requires_and_preserves_raw_pairs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "out"
        child_result = root / "quality.json"
        payload = _claim_payload(kind="quality", pairs=2)
        proc = _invoke(
            out, "quality", _json_writer(child_result, payload),
            "--claim-kind", "quality",
            "--child-result-json", str(child_result),
        )
        assert proc.returncode == 0, proc.stderr
        result = _done(out, "quality")
        assert result["verdict"] == "PASS"
        assert result["measurement_status"] == "measured"
        assert result["claim_details"]["paired_summary"] is None
        assert result["claim_details"]["metadata"]["observation_count"] == 2
        assert result["child_result"]["observations"] == payload["observations"]


def test_child_failed_claim_decision_fails_wrapper() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "out"
        child_result = root / "failed-claim.json"
        proc = _invoke(
            out, "failed-claim", _json_writer(
                child_result, _claim_payload(passed=False)),
            "--claim-kind", "performance",
            "--child-result-json", str(child_result),
        )
        assert proc.returncode == 1
        result = _done(out, "failed-claim")
        assert result["verdict"] == "FAIL"
        assert result["measurement_status"] == "measured"
        assert result["claim_details"]["claim_passed"] is False


def test_raw_stdout_and_stderr_are_preserved_separately() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        stdout = b"out\x00\xff\n"
        stderr = b"err\x00\xfe\n"
        code = (
            "import sys; "
            "sys.stdout.buffer.write(bytes.fromhex(sys.argv[1])); "
            "sys.stdout.buffer.flush(); "
            "sys.stderr.buffer.write(bytes.fromhex(sys.argv[2])); "
            "sys.stderr.buffer.flush()"
        )
        proc = _invoke(
            out, "raw-streams",
            [sys.executable, "-c", code, stdout.hex(), stderr.hex()],
        )
        assert proc.returncode == 0, proc.stderr
        result = _done(out, "raw-streams")
        artifact = _artifact(result)
        assert (artifact / "stdout.raw").read_bytes() == stdout
        assert (artifact / "stderr.raw").read_bytes() == stderr
        assert Path(result["log_path"]).read_bytes() == stdout + stderr
        assert result["stdout_sha256"] == hashlib.sha256(stdout).hexdigest()
        assert result["stderr_sha256"] == hashlib.sha256(stderr).hexdigest()


def test_required_lock_without_live_owner_prevents_child_spawn() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "out"
        marker = root / "child-ran"
        absent_lock = root / "absent.lock"
        code = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('ran')"
        proc = _invoke(
            out, "missing-lock",
            [sys.executable, "-c", code, str(marker)],
            "--lock-path", str(absent_lock), "--require-lock",
        )
        assert proc.returncode == 1
        assert not marker.exists()
        result = _done(out, "missing-lock")
        assert result["verdict"] == "FAIL"
        assert result["exit_code"] is None
        assert "live numeric owner" in result["spawn_error"]
        assert (_artifact(result) / "stdout.raw").read_bytes() == b""


def test_artifact_directory_and_files_publish_atomically_and_uniquely() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        first = _invoke(out, "atomic-a", [sys.executable, "-c", "print('a')"])
        second = _invoke(out, "atomic-b", [sys.executable, "-c", "print('b')"])
        assert first.returncode == second.returncode == 0
        first_result = _done(out, "atomic-a")
        second_result = _done(out, "atomic-b")
        first_artifact = _artifact(first_result)
        second_artifact = _artifact(second_result)
        assert first_artifact != second_artifact
        assert not list(out.glob(".*.artifacts.tmp"))
        assert not list(out.rglob("*.tmp"))
        for result, artifact in (
            (first_result, first_artifact), (second_result, second_artifact),
        ):
            assert result["artifact_manifest_sha256"] == sha256_file_for_test(
                artifact / "manifest.json")


def sha256_file_for_test(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_timing_fields_never_label_wrapper_or_process_time_as_benchmark() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        proc = _invoke(
            out, "timing",
            [sys.executable, "-c", "import time; time.sleep(0.03)"],
        )
        assert proc.returncode == 0, proc.stderr
        result = _done(out, "timing")
        timing = result["timing"]
        assert result["wall_seconds"] == result["wrapper_wall_seconds"]
        assert result["wall_seconds_scope"] == "wrapper_wall_seconds_not_benchmark_time"
        assert timing["wrapper_wall_seconds"] >= timing["child_process_elapsed_seconds"]
        assert timing["setup_before_child_seconds"] > 0
        assert "never a benchmark/model metric" in timing["semantics"]["wrapper_wall_seconds"]
        assert "not a model metric" in timing["semantics"]["child_process_elapsed_seconds"]
        assert "benchmark_seconds" not in result and "benchmark_time" not in result


def test_paired_summary_is_deterministic_and_conservative_for_small_n() -> None:
    observations = _claim_payload(pairs=6)["observations"]
    first = paired_log_ratio_summary(observations, "metrics.total_seconds")
    second = paired_log_ratio_summary(observations, "metrics.total_seconds")
    assert first == second
    assert first["ratio_median"] == 2.0
    assert first["wins"] == 6 and first["losses"] == 0
    assert first["confidence_interval"]["ratio_lower"] == 2.0
    small = paired_log_ratio_summary(
        observations[:5], "metrics.total_seconds")
    assert small["confidence_interval"]["ratio_lower"] is None
    assert "insufficient pairs" in small["confidence_interval"]["note"]


def test_source_change_during_run_fails_gate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        source = Path(tmp) / "source"
        source.mkdir()
        target = source / "probe.py"
        target.write_text("before\n")
        code = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('after\\n')"
        proc = _invoke(
            out, "source-change", [sys.executable, "-c", code, str(target)],
            "--source-root", str(source),
        )
        assert proc.returncode == 1
        result = _done(out, "source-change")
        assert result["verdict"] == "FAIL"
        assert result["source_tree_changed_during_run"] is True
        assert (result["source_manifest"]["tree_sha256"]
                != result["source_manifest_end"]["tree_sha256"])


def test_log_limit_turns_fast_zero_exit_into_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        proc = _invoke(
            out, "log-limit",
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 4096)"],
            "--max-log-bytes", "128",
        )
        assert proc.returncode == 1
        result = _done(out, "log-limit")
        assert result["verdict"] == "FAIL"
        assert result["log_limit_exceeded"] is True
        assert result["log_limit_bytes"] == 128
        assert (_artifact(result) / "stdout.raw").stat().st_size == 4096


def test_oversized_child_result_fails_without_loading_it() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        child_result = out / "large.json"
        code = (
            "import pathlib,sys; "
            "pathlib.Path(sys.argv[1]).write_text('{\"x\":\"' + 'a'*256 + '\"}')"
        )
        proc = _invoke(
            out, "large-child", [sys.executable, "-c", code, str(child_result)],
            "--child-result-json", str(child_result),
            "--max-child-result-bytes", "64",
        )
        assert proc.returncode == 1
        result = _done(out, "large-child")
        assert result["verdict"] == "FAIL"
        assert "limit is 64" in result["child_result_error"]
        assert result["child_result"] is None
        assert not (_artifact(result) / "structured-result.raw.json").exists()


def _run_all() -> None:
    tests = [
        test_backward_compatible_smoke_run_without_result,
        test_nonzero_result,
        test_timeout_records_signal,
        test_legacy_child_result_is_fresh_and_ingested_for_smoke,
        test_measured_claim_records_observations_summary_lock_and_fingerprints,
        test_declared_claim_without_result_fails_as_missing,
        test_missing_requested_child_result_turns_zero_exit_into_failure,
        test_malformed_claim_result_fails_and_preserves_raw_bytes,
        test_explicit_unmeasured_claim_is_distinct_and_nonzero,
        test_measured_quality_claim_requires_and_preserves_raw_pairs,
        test_child_failed_claim_decision_fails_wrapper,
        test_raw_stdout_and_stderr_are_preserved_separately,
        test_required_lock_without_live_owner_prevents_child_spawn,
        test_artifact_directory_and_files_publish_atomically_and_uniquely,
        test_timing_fields_never_label_wrapper_or_process_time_as_benchmark,
        test_paired_summary_is_deterministic_and_conservative_for_small_n,
        test_source_change_during_run_fails_gate,
        test_log_limit_turns_fast_zero_exit_into_failure,
        test_oversized_child_result_fails_without_loading_it,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    try:
        _run_all()
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
