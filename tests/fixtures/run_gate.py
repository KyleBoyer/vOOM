#!/usr/bin/env python3
"""Run one proof command and publish an auditable atomic result envelope.

The parent intentionally imports no ML framework. It captures the child process,
raw stdout/stderr, source and caller-supplied input fingerprints, host pressure,
and environment metadata without resetting MLX counters. ``--claim-kind`` keeps
ordinary smoke wrapping backward compatible while making performance/quality
claims require a structured child result.

Claim-result schema (``voom.gate.observations.v1``)::

  {
    "schema": "voom.gate.observations.v1",
    "claim_kind": "performance",
    "status": "measured",
    "seed": 7,
    "fingerprints": {"model": {...}, "config": {...}},
    "decision": {
      "passed": true, "rule": "lower CI ratio > 1.03",
      "minimum_ratio": 1.03
    },
    "primary_metric": {
      "name": "total_seconds", "path": "metrics.total_seconds",
      "higher_is_better": false
    },
    "observations": [
      {
        "pair_id": "pair-0", "order": ["control", "candidate"],
        "control": {"metrics": {"total_seconds": 2.0}, "tokens": [...]},
        "candidate": {"metrics": {"total_seconds": 1.0}, "tokens": [...]}
      }
    ]
  }

An intentionally unmeasured claim uses the same schema plus
``{"status":"unmeasured", "reason":"..."}``; it is published with an
``UNMEASURED`` verdict and a nonzero wrapper exit. Every measured claim keeps the
complete per-pair objects verbatim. The wrapper additionally derives a
dependency-free, distribution-free paired log-ratio summary for performance.

Practical model-gate pattern (the child must atomically write ``$CHILD_JSON``)::

  RUN_ID=my-gate-$(date +%s)
  CHILD_JSON="$PWD/logs/gates/$RUN_ID.child.json"
  while ! mkdir /tmp/voom-mlx-benchmark.lock 2>/dev/null; do sleep 1; done
  printf '%s\n' "$$" > /tmp/voom-mlx-benchmark.lock/owner
  cleanup_lock() {
    if [ "$(cat /tmp/voom-mlx-benchmark.lock/owner 2>/dev/null)" = "$$" ]; then
      rm -f /tmp/voom-mlx-benchmark.lock/owner
      rmdir /tmp/voom-mlx-benchmark.lock
    fi
  }
  trap cleanup_lock EXIT INT TERM
  caffeinate -is .venv/bin/python tests/fixtures/run_gate.py \
    --result-dir logs/gates --run-id "$RUN_ID" --proof-class E \
    --claim-kind performance --require-lock \
    --fingerprint model_config=/path/to/model/config.json \
    --fingerprint weight_manifest=/path/to/model/voom.safetensors.sha256.json \
    --fingerprint runtime_config=/path/to/run-config.json \
    --child-result-json "$CHILD_JSON" \
    --expected "paired lower confidence bound clears the frozen margin" -- \
    .venv/bin/python experiments/paired_gate.py --result-json "$CHILD_JSON"

``wrapper_wall_seconds`` and its compatibility alias ``wall_seconds`` include
setup and publication work and are explicitly *not* benchmark measurements.
Only metrics supplied in the structured child observations are model metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - the project environment has psutil
    psutil = None


EXCLUDED_DIRS = {
    ".git", ".venv", ".pytest_cache", "__pycache__", "hf_cache", "logs",
    "models", ".kv_prompts", ".kv_spill",
}
SOURCE_SUFFIXES = {".py", ".md", ".json", ".toml", ".yaml", ".yml"}
DEFAULT_MAX_LOG_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_CHILD_RESULT_BYTES = 16 * 1024 * 1024
CHILD_OBSERVATION_SCHEMA = "voom.gate.observations.v1"
ARTIFACT_SCHEMA = "voom.gate.artifacts.v1"
CLAIM_KINDS = ("smoke", "performance", "quality", "performance-quality")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes only after the file and containing directory are durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    atomic_bytes(path, payload)


def atomic_concat(path: Path, sources: list[Path], chunk_bytes: int = 1024 * 1024) -> None:
    """Atomically publish a compatibility log without materializing it in RAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        for source in sources:
            with source.open("rb") as stream:
                while chunk := stream.read(chunk_bytes):
                    view = memoryview(chunk)
                    while view:
                        written = os.write(fd, view)
                        view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    fsync_dir(path.parent)


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a file without materializing an arbitrarily large file in RAM."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(root: Path) -> dict:
    """Hash the relevant local tree deterministically when no git commit exists."""
    files: list[tuple[str, str, int]] = []
    combined = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        rel_text = rel.as_posix()
        files.append((rel_text, digest, size))
        combined.update(rel_text.encode() + b"\0" + digest.encode() + b"\0")
    return {
        "algorithm": "sha256(path\\0sha256\\0)",
        "tree_sha256": combined.hexdigest(),
        "file_count": len(files),
        "files": [{"path": p, "sha256": h, "bytes": n} for p, h, n in files],
    }


def parse_fingerprint_specs(values: list[str], cwd: Path) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("--fingerprint must use LABEL=PATH")
        if label in specs:
            raise ValueError(f"duplicate fingerprint label: {label}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = cwd / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"fingerprint input is not a file: {path}")
        specs[label] = path
    return specs


def input_fingerprints(specs: dict[str, Path]) -> dict:
    result = {}
    for label, path in sorted(specs.items()):
        stat = path.stat()
        result[label] = {
            "path": str(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }
    return result


def package_snapshot() -> dict:
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name.lower()] = dist.version
    ordered = dict(sorted(packages.items()))
    payload = json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "packages": ordered}


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_snapshot() -> dict:
    total_memory = None
    if psutil is not None:
        total_memory = psutil.virtual_memory().total
    packages = package_snapshot()
    return {
        # Original v1 fields remain for readers that consume them directly.
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        # Explicitly named v2 metadata avoids parsing free-form strings.
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "mlx": {"distribution_version": distribution_version("mlx")},
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": total_memory,
        },
    }


def lock_snapshot(path: Path) -> dict:
    path = path.expanduser().resolve()
    exists = path.exists()
    owner_path = path / "owner" if path.is_dir() else path
    owner = None
    owner_error = None
    if exists:
        try:
            owner = owner_path.read_text(errors="replace")[:4096].strip() or None
        except OSError as exc:
            owner_error = f"{type(exc).__name__}: {exc}"
    owner_pid = int(owner) if owner and owner.isdecimal() else None
    owner_alive = None
    if owner_pid is not None:
        try:
            os.kill(owner_pid, 0)
            owner_alive = True
        except ProcessLookupError:
            owner_alive = False
        except PermissionError:
            owner_alive = True
    stat = path.stat() if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "owner_path": str(owner_path),
        "owner": owner,
        "owner_pid": owner_pid,
        "owner_alive": owner_alive,
        "owner_error": owner_error,
        "inode": stat.st_ino if stat is not None else None,
        "mtime_ns": stat.st_mtime_ns if stat is not None else None,
    }


def pressure_sample(child_pid: int | None, root: Path) -> dict:
    sample = {
        "root_free_bytes": shutil.disk_usage("/").free,
        "workspace_free_bytes": shutil.disk_usage(root).free,
        "swap_free_bytes": None,
        "system_available_bytes": None,
        "child_tree_rss_bytes": None,
    }
    if psutil is None:
        return sample
    sample["swap_free_bytes"] = psutil.swap_memory().free
    sample["system_available_bytes"] = psutil.virtual_memory().available
    if child_pid is not None:
        try:
            proc = psutil.Process(child_pid)
            members = [proc] + proc.children(recursive=True)
            sample["child_tree_rss_bytes"] = sum(
                process.memory_info().rss
                for process in members if process.is_running()
            )
        except (psutil.Error, ProcessLookupError):
            pass
    return sample


def update_extrema(extrema: dict, sample: dict) -> None:
    for key in (
        "root_free_bytes", "workspace_free_bytes", "swap_free_bytes",
        "system_available_bytes",
    ):
        value = sample.get(key)
        if value is not None:
            extrema[key.replace("_bytes", "_min_bytes")] = min(
                extrema.get(key.replace("_bytes", "_min_bytes"), value), value
            )
    rss = sample.get("child_tree_rss_bytes")
    if rss is not None:
        extrema["child_tree_rss_max_bytes"] = max(
            extrema.get("child_tree_rss_max_bytes", rss), rss
        )


def nested_value(value: dict, dotted_path: str):
    current = value
    for component in dotted_path.split("."):
        if not isinstance(current, dict) or component not in current:
            raise KeyError(dotted_path)
        current = current[component]
    return current


def paired_log_ratio_summary(
    observations: list[dict],
    metric_path: str,
    *,
    higher_is_better: bool = False,
    confidence: float = 0.95,
) -> dict:
    """Summarize paired measurements with a conservative median interval.

    Positive log ratios always favor the candidate. The confidence interval is
    the narrowest distribution-free sign/order-statistic interval whose achieved
    coverage is at least ``confidence``. With fewer than six pairs a two-sided
    95% finite interval is impossible, so the bounds are reported as ``None``.
    """
    if not observations:
        raise ValueError("paired observations must not be empty")
    if not metric_path:
        raise ValueError("metric_path must not be empty")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")

    log_ratios = []
    for index, observation in enumerate(observations):
        try:
            control = nested_value(observation["control"], metric_path)
            candidate = nested_value(observation["candidate"], metric_path)
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"observation {index} lacks numeric paired metric {metric_path!r}"
            ) from exc
        if (isinstance(control, bool) or isinstance(candidate, bool)
                or not isinstance(control, (int, float))
                or not isinstance(candidate, (int, float))):
            raise ValueError(f"observation {index} paired metrics must be numeric")
        control = float(control)
        candidate = float(candidate)
        if (not math.isfinite(control) or not math.isfinite(candidate)
                or control <= 0 or candidate <= 0):
            raise ValueError(
                f"observation {index} paired metrics must be finite and positive"
            )
        ratio = candidate / control if higher_is_better else control / candidate
        log_ratios.append(math.log(ratio))

    ordered = sorted(log_ratios)
    median_log_ratio = statistics.median(ordered)
    lower_index = None
    achieved_confidence = None
    for candidate_index in range((len(ordered) - 1) // 2 + 1):
        tail = sum(
            math.comb(len(ordered), count)
            for count in range(candidate_index + 1)
        ) / (2 ** len(ordered))
        coverage = 1 - 2 * tail
        if coverage >= confidence:
            lower_index = candidate_index
            achieved_confidence = coverage
        else:
            break

    if lower_index is None:
        interval = {
            "method": "distribution-free-median-order-statistic",
            "requested_confidence": confidence,
            "achieved_confidence": None,
            "log_ratio_lower": None,
            "log_ratio_upper": None,
            "ratio_lower": None,
            "ratio_upper": None,
            "note": "insufficient pairs for a finite interval at requested confidence",
        }
    else:
        lower = ordered[lower_index]
        upper = ordered[len(ordered) - lower_index - 1]
        interval = {
            "method": "distribution-free-median-order-statistic",
            "requested_confidence": confidence,
            "achieved_confidence": achieved_confidence,
            "log_ratio_lower": lower,
            "log_ratio_upper": upper,
            "ratio_lower": math.exp(lower),
            "ratio_upper": math.exp(upper),
            "note": None,
        }

    return {
        "pairs": len(log_ratios),
        "metric_path": metric_path,
        "higher_is_better": higher_is_better,
        "log_ratios": log_ratios,
        "log_ratio_median": median_log_ratio,
        "ratio_median": math.exp(median_log_ratio),
        "wins": sum(value > 0 for value in log_ratios),
        "ties": sum(value == 0 for value in log_ratios),
        "losses": sum(value < 0 for value in log_ratios),
        "confidence_interval": interval,
    }


def validate_claim_result(value: dict, declared_kind: str) -> dict:
    """Validate and normalize one structured performance/quality result."""
    if value.get("schema") != CHILD_OBSERVATION_SCHEMA:
        raise ValueError(
            f"claim result schema must be {CHILD_OBSERVATION_SCHEMA!r}"
        )
    if value.get("claim_kind") != declared_kind:
        raise ValueError(
            f"child claim_kind {value.get('claim_kind')!r} does not match "
            f"declared {declared_kind!r}"
        )
    status = value.get("status")
    if status == "unmeasured":
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("unmeasured claim result requires a nonempty reason")
        return {
            "status": "unmeasured",
            "reason": reason,
            "claim_passed": None,
            "metadata": {
                "seed": value.get("seed"),
                "fingerprints": value.get("fingerprints"),
                "order": [],
                "observation_count": 0,
            },
            "paired_summary": None,
        }
    if status != "measured":
        raise ValueError("claim result status must be 'measured' or 'unmeasured'")

    if "seed" not in value or isinstance(value["seed"], (dict, list)):
        raise ValueError("measured claim result requires an explicit scalar seed")
    fingerprints = value.get("fingerprints")
    if (not isinstance(fingerprints, dict)
            or not fingerprints.get("model")
            or not fingerprints.get("config")):
        raise ValueError(
            "measured claim result requires model and config fingerprints"
        )
    decision = value.get("decision")
    if (not isinstance(decision, dict)
            or not isinstance(decision.get("passed"), bool)
            or not isinstance(decision.get("rule"), str)
            or not decision["rule"].strip()):
        raise ValueError(
            "measured claim result requires decision.passed and a nonempty rule"
        )
    observations = value.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("measured claim result requires raw paired observations")

    pair_ids = set()
    orders = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise ValueError(f"observation {index} must be an object")
        pair_id = observation.get("pair_id")
        if not isinstance(pair_id, (str, int)) or isinstance(pair_id, bool):
            raise ValueError(f"observation {index} requires a string/integer pair_id")
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair_id: {pair_id!r}")
        pair_ids.add(pair_id)
        order = observation.get("order")
        if (not isinstance(order, list) or len(order) != 2
                or set(order) != {"control", "candidate"}):
            raise ValueError(
                f"observation {index} order must contain control and candidate once"
            )
        if not isinstance(observation.get("control"), dict):
            raise ValueError(f"observation {index} control must be an object")
        if not isinstance(observation.get("candidate"), dict):
            raise ValueError(f"observation {index} candidate must be an object")
        orders.append(order)

    paired_summary = None
    if "performance" in declared_kind:
        primary = value.get("primary_metric")
        if (not isinstance(primary, dict)
                or not isinstance(primary.get("name"), str)
                or not primary["name"].strip()
                or not isinstance(primary.get("path"), str)
                or not primary["path"].strip()
                or not isinstance(primary.get("higher_is_better"), bool)):
            raise ValueError(
                "performance claim requires primary_metric name/path/"
                "higher_is_better"
            )
        paired_summary = paired_log_ratio_summary(
            observations,
            primary["path"],
            higher_is_better=primary["higher_is_better"],
        )
        paired_summary["metric_name"] = primary["name"]
        minimum_ratio = decision.get("minimum_ratio")
        if (isinstance(minimum_ratio, bool)
                or not isinstance(minimum_ratio, (int, float))
                or not math.isfinite(float(minimum_ratio))
                or minimum_ratio <= 0):
            raise ValueError(
                "performance claim requires a finite positive "
                "decision.minimum_ratio"
            )
        lower = paired_summary["confidence_interval"]["ratio_lower"]
        evidence_passed = lower is not None and lower > float(minimum_ratio)
        paired_summary["decision"] = {
            "minimum_ratio": float(minimum_ratio),
            "lower_bound_passed": evidence_passed,
        }
        if decision["passed"] and not evidence_passed:
            raise ValueError(
                "claim decision says PASS but paired lower confidence bound does "
                "not clear decision.minimum_ratio"
            )

    return {
        "status": "measured",
        "reason": None,
        "claim_passed": decision["passed"],
        "decision_rule": decision["rule"],
        "metadata": {
            "seed": value["seed"],
            "fingerprints": fingerprints,
            "order": orders,
            "observation_count": len(observations),
        },
        "paired_summary": paired_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--proof-class", choices=("L0", "L1", "E"), required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument(
        "--claim-kind", choices=CLAIM_KINDS, default="smoke",
        help="smoke permits no child result; other kinds require schema-validated data",
    )
    parser.add_argument("--timeout", type=float, default=0.0,
                        help="child-process seconds; 0 disables timeout")
    parser.add_argument("--kill-grace", type=float, default=5.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--max-log-bytes", type=int, default=DEFAULT_MAX_LOG_BYTES,
        help="terminate/fail if combined raw streams exceed this size; 0 disables",
    )
    parser.add_argument(
        "--max-child-result-bytes", type=int,
        default=DEFAULT_MAX_CHILD_RESULT_BYTES,
        help="reject a child JSON artifact larger than this; 0 disables",
    )
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument(
        "--child-result-json", type=Path,
        help="fresh JSON object the child must create; relative paths use --cwd",
    )
    parser.add_argument(
        "--fingerprint", action="append", default=[], metavar="LABEL=PATH",
        help="hash a model manifest/config/input file without loading model weights",
    )
    parser.add_argument(
        "--lock-path", type=Path,
        default=Path("/tmp/voom-mlx-benchmark.lock"),
    )
    parser.add_argument(
        "--require-lock", action="store_true",
        help="fail before spawning unless --lock-path exists with a readable owner",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("child command is required after --")
    if (args.timeout < 0 or args.kill_grace < 0 or args.poll_seconds <= 0
            or args.max_log_bytes < 0 or args.max_child_result_bytes < 0):
        parser.error(
            "timeout/grace/byte limits must be nonnegative and poll-seconds positive"
        )
    return args


def terminate_group(child: subprocess.Popen, grace: float) -> None:
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def output_bytes(stdout_path: Path, stderr_path: Path) -> int:
    return sum(
        path.stat().st_size if path.exists() else 0
        for path in (stdout_path, stderr_path)
    )


def main() -> int:
    wrapper_started_perf = time.perf_counter()
    wrapper_started_at = utc_now()
    args = parse_args()
    run_id = args.run_id or f"gate-{uuid.uuid4().hex}"
    if "/" in run_id or run_id in {".", ".."}:
        raise SystemExit("run-id must be a filename-safe component")

    result_dir = args.result_dir.expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    running_path = result_dir / f"{run_id}.running.json"
    done_path = result_dir / f"{run_id}.done.json"
    log_path = result_dir / f"{run_id}.log"
    if running_path.exists() or done_path.exists():
        raise SystemExit(f"refusing to overwrite existing run artifact: {run_id}")

    artifact_id = f"{run_id}.{uuid.uuid4().hex}"
    artifact_staging = result_dir / f".{artifact_id}.artifacts.tmp"
    artifact_dir = result_dir / f"{artifact_id}.artifacts"
    stdout_path = artifact_staging / "stdout.raw"
    stderr_path = artifact_staging / "stderr.raw"

    child_result_path = None
    if args.child_result_json is not None:
        child_result_path = args.child_result_json.expanduser()
        if not child_result_path.is_absolute():
            child_result_path = args.cwd / child_result_path
        child_result_path = child_result_path.resolve()
        if child_result_path.exists():
            raise SystemExit(
                f"refusing stale child-result artifact: {child_result_path}"
            )

    metadata = {}
    if args.metadata_json:
        metadata = json.loads(args.metadata_json.read_text())
        if not isinstance(metadata, dict):
            raise SystemExit("metadata-json must contain one JSON object")

    try:
        fingerprint_specs = parse_fingerprint_specs(args.fingerprint, args.cwd)
        fingerprints = input_fingerprints(fingerprint_specs)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    lock = lock_snapshot(args.lock_path)
    artifact_staging.mkdir(mode=0o755)
    base = {
        "schema": "vmodel.gate.v1",
        "semantics_version": 2,
        "run_id": run_id,
        "artifact_id": artifact_id,
        "state": "running",
        "started_at": wrapper_started_at,
        "parent_pid": os.getpid(),
        "child_pid": None,
        "command": args.command,
        "cwd": str(args.cwd.resolve()),
        "proof_class": args.proof_class,
        "expected_gate": args.expected,
        "run_mode": "smoke" if args.claim_kind == "smoke" else "claim",
        "claim_kind": args.claim_kind,
        "metadata": metadata,
        "input_fingerprints": fingerprints,
        "lock": lock,
        "artifact_staging_dir": str(artifact_staging),
        "artifact_dir": str(artifact_dir),
    }
    atomic_json(running_path, base)

    manifest = source_manifest(args.source_root.resolve())
    environment = environment_snapshot()
    extrema: dict = {}
    timed_out = False
    log_limit_exceeded = False
    spawn_error = None
    child = None
    child_started_at = None
    child_finished_at = None
    child_started_perf = None
    child_process_elapsed = None

    lock_before_child = lock_snapshot(args.lock_path)
    lock_changed_before_child = any(
        lock_before_child.get(key) != lock.get(key)
        for key in ("exists", "owner", "inode")
    )
    base["lock_before_child"] = lock_before_child
    base["lock_changed_before_child"] = lock_changed_before_child
    atomic_json(running_path, base)
    if args.require_lock and (
        not lock_before_child["exists"]
        or lock_before_child["owner_pid"] is None
        or not lock_before_child["owner_alive"]
        or lock_changed_before_child
    ):
        spawn_error = "required benchmark lock lacks a live numeric owner"

    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            try:
                if spawn_error is None:
                    child_started_at = utc_now()
                    child_started_perf = time.perf_counter()
                    child = subprocess.Popen(
                        args.command,
                        cwd=args.cwd,
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=True,
                    )
                    base["child_pid"] = child.pid
                    base["child_started_at"] = child_started_at
                    if psutil is not None:
                        try:
                            base["child_create_time"] = (
                                psutil.Process(child.pid).create_time()
                            )
                        except psutil.Error:
                            base["child_create_time"] = None
                    atomic_json(running_path, base)
                    while child.poll() is None:
                        update_extrema(
                            extrema, pressure_sample(child.pid, args.source_root)
                        )
                        if (args.max_log_bytes
                                and output_bytes(stdout_path, stderr_path)
                                > args.max_log_bytes):
                            log_limit_exceeded = True
                            terminate_group(child, args.kill_grace)
                            break
                        elapsed = time.perf_counter() - child_started_perf
                        if args.timeout and elapsed >= args.timeout:
                            timed_out = True
                            terminate_group(child, args.kill_grace)
                            break
                        wait_seconds = args.poll_seconds
                        if args.timeout:
                            wait_seconds = min(
                                wait_seconds, max(0.001, args.timeout - elapsed)
                            )
                        try:
                            child.wait(timeout=wait_seconds)
                        except subprocess.TimeoutExpired:
                            pass
                    child.wait()
                    child_finished_at = utc_now()
                    child_process_elapsed = (
                        time.perf_counter() - child_started_perf
                    )
                    if (args.max_log_bytes
                            and output_bytes(stdout_path, stderr_path)
                            > args.max_log_bytes):
                        log_limit_exceeded = True
                    update_extrema(
                        extrema, pressure_sample(child.pid, args.source_root)
                    )
            except Exception as exc:
                spawn_error = f"{type(exc).__name__}: {exc}"
                if child is not None:
                    terminate_group(child, args.kill_grace)
                    child_finished_at = utc_now()
                    if child_started_perf is not None:
                        child_process_elapsed = (
                            time.perf_counter() - child_started_perf
                        )
            finally:
                stdout.flush()
                stderr.flush()
                os.fsync(stdout.fileno())
                os.fsync(stderr.fileno())
    except Exception as exc:
        spawn_error = spawn_error or f"{type(exc).__name__}: {exc}"

    returncode = child.returncode if child is not None else None
    atomic_concat(log_path, [stdout_path, stderr_path])

    child_result = None
    child_result_payload = None
    child_result_error = None
    child_result_sha256 = None
    measurement_status = (
        "not_requested" if args.claim_kind == "smoke" else "missing"
    )
    claim_details = None
    if child_result_path is not None:
        try:
            result_size = child_result_path.stat().st_size
            if (args.max_child_result_bytes
                    and result_size > args.max_child_result_bytes):
                raise ValueError(
                    f"child result is {result_size} bytes, limit is "
                    f"{args.max_child_result_bytes}"
                )
            child_result_payload = child_result_path.read_bytes()
            child_result_sha256 = hashlib.sha256(child_result_payload).hexdigest()
            child_result = json.loads(child_result_payload)
            if not isinstance(child_result, dict):
                raise TypeError("child result must be one JSON object")
            if args.claim_kind == "smoke":
                measurement_status = "provided_unvalidated"
            else:
                claim_details = validate_claim_result(
                    child_result, args.claim_kind
                )
                measurement_status = claim_details["status"]
        except FileNotFoundError as exc:
            child_result_error = f"{type(exc).__name__}: {exc}"
            measurement_status = "missing"
        except Exception as exc:
            child_result_error = f"{type(exc).__name__}: {exc}"
            measurement_status = "malformed"
    elif args.claim_kind != "smoke":
        child_result_error = (
            "ValueError: declared performance/quality claim requires "
            "--child-result-json"
        )

    structured_snapshot_path = None
    if child_result_payload is not None:
        structured_snapshot_path = artifact_staging / "structured-result.raw.json"
        atomic_bytes(structured_snapshot_path, child_result_payload)

    try:
        fingerprints_end = input_fingerprints(fingerprint_specs)
        input_fingerprints_changed = fingerprints_end != fingerprints
    except OSError as exc:
        fingerprints_end = None
        input_fingerprints_changed = True
        child_result_error = child_result_error or (
            f"input fingerprint disappeared during run: {type(exc).__name__}: {exc}"
        )

    manifest_end = source_manifest(args.source_root.resolve())
    source_tree_changed = (
        manifest_end["tree_sha256"] != manifest["tree_sha256"]
    )
    lock_end = lock_snapshot(args.lock_path)
    lock_changed = any(
        lock_end.get(key) != lock_before_child.get(key)
        for key in ("exists", "owner", "inode")
    )
    lock_error = None
    if args.require_lock and (
        not lock_end["exists"] or lock_end["owner_pid"] is None
        or not lock_end["owner_alive"] or lock_changed
    ):
        lock_error = "required benchmark lock changed or disappeared during child run"

    stdout_info = {
        "file": "stdout.raw",
        "bytes": stdout_path.stat().st_size,
        "sha256": sha256_file(stdout_path),
    }
    stderr_info = {
        "file": "stderr.raw",
        "bytes": stderr_path.stat().st_size,
        "sha256": sha256_file(stderr_path),
    }
    structured_info = None
    if structured_snapshot_path is not None:
        structured_info = {
            "file": structured_snapshot_path.name,
            "bytes": structured_snapshot_path.stat().st_size,
            "sha256": sha256_file(structured_snapshot_path),
        }
    artifact_manifest = {
        "schema": ARTIFACT_SCHEMA,
        "artifact_id": artifact_id,
        "run_id": run_id,
        "created_at": utc_now(),
        "command": args.command,
        "claim_kind": args.claim_kind,
        "measurement_status": measurement_status,
        "exit_code": returncode if returncode is not None and returncode >= 0 else None,
        "signal": -returncode if returncode is not None and returncode < 0 else None,
        "stdout": stdout_info,
        "stderr": stderr_info,
        "structured_result": structured_info,
    }
    artifact_manifest_path = artifact_staging / "manifest.json"
    atomic_json(artifact_manifest_path, artifact_manifest)

    artifact_error = None
    try:
        fsync_dir(artifact_staging)
        os.replace(artifact_staging, artifact_dir)
        fsync_dir(result_dir)
    except Exception as exc:
        artifact_error = f"{type(exc).__name__}: {exc}"

    operational_pass = (
        returncode == 0
        and not timed_out
        and not log_limit_exceeded
        and spawn_error is None
        and child_result_error is None
        and artifact_error is None
        and lock_error is None
        and not source_tree_changed
        and not input_fingerprints_changed
    )
    if not operational_pass:
        verdict = "FAIL"
    elif args.claim_kind != "smoke" and measurement_status == "unmeasured":
        verdict = "UNMEASURED"
    elif (args.claim_kind != "smoke"
          and (claim_details is None or not claim_details["claim_passed"])):
        verdict = "FAIL"
    else:
        verdict = "PASS"

    wrapper_wall_seconds = time.perf_counter() - wrapper_started_perf
    timing = {
        "wrapper_wall_seconds": wrapper_wall_seconds,
        "setup_before_child_seconds": (
            child_started_perf - wrapper_started_perf
            if child_started_perf is not None else None
        ),
        "child_process_elapsed_seconds": child_process_elapsed,
        "child_started_at": child_started_at,
        "child_finished_at": child_finished_at,
        "poll_interval_seconds": args.poll_seconds,
        "semantics": {
            "wrapper_wall_seconds": (
                "wrapper setup + child process observation + polling + artifact "
                "publication; never a benchmark/model metric"
            ),
            "child_process_elapsed_seconds": (
                "spawn-to-observed-exit elapsed time; process scope, not a model "
                "metric; use structured child observations for benchmark metrics"
            ),
        },
    }

    done = dict(base)
    done.update({
        "state": "done",
        "finished_at": utc_now(),
        # Backward-compatible alias, now carrying an explicit non-benchmark scope.
        "wall_seconds": wrapper_wall_seconds,
        "wall_seconds_scope": "wrapper_wall_seconds_not_benchmark_time",
        "wrapper_wall_seconds": wrapper_wall_seconds,
        "timing": timing,
        "verdict": verdict,
        "timed_out": timed_out,
        "log_limit_bytes": args.max_log_bytes,
        "log_limit_exceeded": log_limit_exceeded,
        "child_result_limit_bytes": args.max_child_result_bytes,
        "exit_code": returncode if returncode is not None and returncode >= 0 else None,
        "signal": -returncode if returncode is not None and returncode < 0 else None,
        "spawn_error": spawn_error,
        "child_result_path": str(child_result_path) if child_result_path else None,
        "child_result_sha256": child_result_sha256,
        "child_result_error": child_result_error,
        "child_result": child_result,
        "measurement_status": measurement_status,
        "claim_details": claim_details,
        "pressure": extrema,
        "source_manifest": manifest,
        "source_manifest_end": manifest_end,
        "source_tree_changed_during_run": source_tree_changed,
        "input_fingerprints_end": fingerprints_end,
        "input_fingerprints_changed_during_run": input_fingerprints_changed,
        "lock_end": lock_end,
        "lock_changed_during_run": lock_changed,
        "lock_error": lock_error,
        "environment": environment,
        "artifact_staging_dir": (
            None if artifact_error is None else str(artifact_staging)
        ),
        "artifact_dir": str(artifact_dir),
        "artifact_error": artifact_error,
        "artifact_manifest_sha256": (
            sha256_file(artifact_dir / "manifest.json")
            if artifact_error is None else None
        ),
        "stdout_path": (
            str(artifact_dir / "stdout.raw") if artifact_error is None else None
        ),
        "stdout_sha256": stdout_info["sha256"],
        "stderr_path": (
            str(artifact_dir / "stderr.raw") if artifact_error is None else None
        ),
        "stderr_sha256": stderr_info["sha256"],
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "log_compatibility_layout": (
            "stdout bytes followed by stderr bytes; use raw artifact streams when "
            "stream identity matters"
        ),
        "metal_peak_bytes": None,
        "metal_peak_note": (
            "parent does not import MLX; structured child observations must provide "
            "request-local Metal metrics"
        ),
    })
    atomic_json(done_path, done)
    running_path.unlink(missing_ok=True)
    fsync_dir(result_dir)
    print(f"{verdict} {done_path} child_returncode={returncode}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
