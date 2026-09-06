#!/usr/bin/env python3
"""Private first-token HTTP endpoint diagnostic, never a performance benchmark.

Start this instead of ``python -m runtime.server`` after a fresh preflight,
then send exactly one existing HTTP replay with max_output_tokens=1. The
normal server applies the requested profiles and owns the engine/lock. This
wrapper normally observes its prepared tokens and returned endpoint without changing
request arguments, model operations, or the returned result. An explicit
--diagnose-retained-prefix experiment may replace retained PLE buffers with
bit-identical independent storage AFTER generation, in this one-call server only.
Hashing copies
state to the host and can affect pressure/latency: both costs are separated.
No prompt, raw text, token IDs, logits, or tensor payloads are published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "voom.qwen4-hot-boundary-http-first-token-diagnostic.v1"


def _token_digest(values) -> dict:
    if not isinstance(values, (list, tuple)) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values):
        raise ValueError("actual nonnegative integer token IDs are required")
    encoded = json.dumps(list(values), separators=(",", ":")).encode("ascii")
    return {"count": len(values), "sha256": hashlib.sha256(encoded).hexdigest(),
            "encoding": "compact-json-integer-array-v1"}


def _hidden_digest(value, *, array_module, numpy_module) -> dict:
    """Read existing BF16 bits; never cast values or recompute model operators."""
    if value is None or value.dtype != array_module.bfloat16:
        raise ValueError("the Qwen4 endpoint must expose BF16 _h_last")
    if (len(value.shape) < 3 or tuple(value.shape[:2]) != (1, 1)
            or any(int(size) <= 0 for size in value.shape)):
        raise ValueError("_h_last must contain exactly one nonempty endpoint row")
    host = numpy_module.asarray(value.view(array_module.uint16))
    raw = host.tobytes(order="C")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "shape": list(host.shape), "dtype": str(value.dtype),
            "encoding": "bf16-bit-view-uint16-c-order-v1"}


def _atomic_write_private(path: Path, document: dict) -> None:
    """Publish a complete 0600 artifact atomically; never overwrite a result."""
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link publishes fully-written bytes while
        # atomically refusing an existing destination (unlike os.replace).
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


class FirstTokenHTTPProbe:
    """Dependency-injected observer; module import itself never imports MLX."""

    def __init__(self, original, *, artifact, expected_prompt_tokens, label,
                 engine_type, state_digest, hidden_digest, pressure,
                 metal_memory, profile_identity, model_revision,
                 clock=time.perf_counter, publish=_atomic_write_private,
                 retained_diagnostic=None):
        if expected_prompt_tokens <= 0:
            raise ValueError("expected prompt-token count must be positive")
        self.original = original
        self.artifact = Path(artifact)
        self.expected_prompt_tokens = expected_prompt_tokens
        self.label = label
        self.engine_type = engine_type
        self.state_digest = state_digest
        self.hidden_digest = hidden_digest
        self.pressure = pressure
        self.metal_memory = metal_memory
        self.profile_identity = profile_identity
        self.model_revision = model_revision
        self.clock = clock
        self.publish = publish
        self.claimed = False
        self.retained_diagnostic = retained_diagnostic

    def _publish(self, document):
        try:
            self.publish(self.artifact, document)
        except Exception as error:
            # Missing/unavailable artifact fails the caller's diagnostic gate,
            # but must not replace a generation failure or a valid response.
            try:
                print("[qwen4-boundary-probe] artifact_write_failed "
                      f"error_type={type(error).__name__}", flush=True)
            except Exception:
                pass

    def __call__(self, engine, *args, **kwargs):
        if self.claimed:
            raise RuntimeError("first-token probe permits exactly one generation")
        self.claimed = True
        started = self.clock()
        document = {
            "schema": SCHEMA, "available": False, "label": self.label,
            "scope": "one_http_engine_generation_first_token_endpoint",
            "timing_benchmark": False, "quality_benchmark": False,
            "full_harness_proof": False, "first_token_only": True,
        }
        phase = "request_validation"
        try:
            if type(engine) is not self.engine_type:
                raise ValueError("requires the concrete Qwen4 MTP wrapper")
            target = engine.target
            if getattr(getattr(target, "cfg", None), "model_type", None) != "qwen4_exp":
                raise ValueError("requires a Qwen4 endpoint owner")
            prompt = args[0] if args else kwargs.get("prompt")
            max_tokens = args[1] if len(args) > 1 else kwargs.get("max_tokens")
            if (isinstance(max_tokens, bool) or not isinstance(max_tokens, int)
                    or max_tokens != 1):
                raise ValueError("first-token probe requires max_tokens=1")
            prepared = _token_digest(getattr(prompt, "token_ids", None))
            if prepared["count"] != self.expected_prompt_tokens:
                raise ValueError("unexpected prepared prompt-token count")
            boundary = getattr(prompt, "stable_boundary_tokens", 0)
            if (isinstance(boundary, bool) or not isinstance(boundary, int)
                    or not 0 <= boundary < prepared["count"]):
                raise ValueError("invalid stable-boundary metadata")
            document.update({
                "prepared_tokens": prepared, "stable_boundary_tokens": boundary,
                "max_output_tokens": 1,
                "runtime_profile_identity": self.profile_identity(),
                "model_revision": self.model_revision(Path(target._model_dir)),
                "pressure_before_generation": self.pressure(),
                "metal_before_generation": self.metal_memory(),
            })
            generate_started = self.clock()
            phase = "generation"
            # Forward the original objects/arguments unchanged, exactly once.
            result = self.original(engine, *args, **kwargs)
            generate_finished = self.clock()
        except BaseException as error:
            document.update({"failure_phase": phase, "error_type": type(error).__name__})
            self._publish(document)
            raise

        phase = "endpoint_capture"
        capture_started = self.clock()
        try:
            document["pressure_after_generation"] = self.pressure()
            document["metal_after_generation"] = self.metal_memory()
            generated = _token_digest(result.get("tokens"))
            if generated["count"] != 1:
                raise ValueError("generation did not return exactly one actual token")
            kv = getattr(target, "last_kv", None)
            hidden = getattr(target, "_h_last", None)
            if (kv is None or getattr(kv, "kda_cache", None) is None
                    or getattr(kv, "qwen4_cache", None) is None
                    or int(kv.offset) != prepared["count"]):
                raise ValueError("missing or misaligned Qwen4 first-token endpoint")
            if int(result.get("kv_positions", -1)) != int(kv.offset):
                raise ValueError("result and authoritative endpoint positions differ")
            if self.retained_diagnostic is not None:
                # This opt-in experiment must precede ordinary host hashing,
                # which could materialize the retained-view hypothesis away.
                document["retained_fork_diagnostic"] = self.retained_diagnostic(
                    target, prompt.token_ids)
            digest, arrays, payload, components = self.state_digest(kv)
            if arrays <= 0 or payload <= 0:
                raise ValueError("Qwen4 endpoint digest contains no state arrays")
            hidden_witness = self.hidden_digest(hidden)
            # Only scalar/hash metadata leaves this scope. Do not keep the
            # endpoint, hidden array, or temporary host copies in the observer.
            state = {"sha256": digest, "arrays": int(arrays), "bytes": int(payload),
                     "components": components, "positions": int(kv.offset)}
            del kv, hidden
            stats = result.get("path_stats") or {}
            raw_text = result.get("text")
            if not isinstance(raw_text, str):
                raise ValueError("raw engine text is unavailable")
            raw_bytes = raw_text.encode("utf-8")
            document.update({
                "generated_tokens": generated,
                "engine_text": {"bytes": len(raw_bytes),
                                "sha256": hashlib.sha256(raw_bytes).hexdigest()},
                "state": state, "hidden": hidden_witness,
                "generation": {
                    key: result.get(key) for key in (
                        "prefill_s", "decode_s", "total_s", "kv_bytes",
                        "kv_positions", "true_peak_metal_bytes", "termination_reason")
                },
                "path": {key: stats.get(key) for key in (
                    "prompt_cache_source", "prompt_cache_prefix_tokens",
                    "prompt_cache_exact_hit", "hot_prompt_boundary_fork_tokens",
                    "hot_prompt_boundary_matched_fork", "hot_prompt_boundary_layer_stationary",
                    "qwen4_hot_boundary_requested", "qwen4_hot_boundary_effective",
                    "qwen4_hot_boundary_tile", "qwen4_hot_boundary_eligible",
                    "qwen4_hot_boundary_policy_eligible",
                    "qwen4_hot_boundary_reason",
                    "prefill_step_size", "qwen_native_fused_delta_prefill",
                    "qwen_compiled_delta_prefill", "qwen4_mtp_used",
                    "qwen4_mtp_fallback_reason", "memory_prefill_retries")},
                "pressure_after_instrumentation": self.pressure(),
                "metal_after_instrumentation": self.metal_memory(),
                "available": True,
            })
        except Exception as error:
            document.update({"available": False, "failure_phase": phase,
                             "error_type": type(error).__name__})
        finally:
            capture_finished = self.clock()
            document["instrumentation"] = {
                "pre_generation_seconds": generate_started - started,
                "generation_invocation_seconds": generate_finished - generate_started,
                "endpoint_capture_seconds": capture_finished - capture_started,
                "artifact_write_included": False,
                "state_host_read_bytes": (document.get("state") or {}).get("bytes"),
                "hidden_host_read_bytes": (document.get("hidden") or {}).get("bytes"),
                "note": "Hashing adds host reads/synchronization; not a serving timing or pressure proof.",
            }
            self._publish(document)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expected-prompt-tokens", type=int, required=True)
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--profile-dir", action="append", default=[])
    parser.add_argument("--diagnose-retained-prefix", type=int, default=0,
                        help="Post-generation RAM-fork materialization/PLE-detachment experiment; max1 only")
    parser.add_argument("--diagnose-retained-kda-conv", action="store_true",
                        help="Also detach retained DeltaNet convolution histories; requires retained-prefix diagnostic")
    args = parser.parse_args(argv)
    if args.artifact.exists() or args.artifact.is_symlink():
        parser.error("artifact already exists; choose a fresh private artifact path")
    if args.expected_prompt_tokens <= 0 or not 1 <= args.port <= 65535:
        parser.error("positive expected prompt-token count and valid port are required")
    if not 0 <= args.diagnose_retained_prefix < args.expected_prompt_tokens:
        parser.error("retained-prefix diagnostic must be a strict nonnegative prompt prefix")
    if args.diagnose_retained_kda_conv and not args.diagnose_retained_prefix:
        parser.error("KDA convolution diagnostic requires a retained prefix")

    # Heavy imports occur only in this explicit CLI entry point, after the
    # caller's preflight. Importing/testing schema helpers never initializes MLX.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import mlx.core as mx
    import numpy as np
    from runtime import server
    from runtime.profiles import active_runtime_profile_fields
    from runtime.qwen4_mtp import Qwen4MTPSpeculativeEngine
    from tests.fixtures.qwen4_flash_next_real_oracle import (
        _model_revision, _pressure, _state_digest)

    def array_digest(value):
        host = np.asarray(value.view(mx.uint16)
                          if value.dtype == mx.bfloat16 else value)
        return {"sha256": hashlib.sha256(host.tobytes(order="C")).hexdigest(),
                "shape": list(value.shape), "dtype": str(value.dtype)}

    retained_diagnostic = None
    if args.diagnose_retained_prefix:
        import psutil
        from runtime.kv_cache import KVCache
        from tests.fixtures.qwen4_retained_fork_diagnostic import (
            complete_state_metadata, copy_ple_bits, diagnose_retained_fork)

        def synchronize_ple(values):
            mx.eval(*[value for value in values if value is not None])
            mx.synchronize()

        def retained_diagnostic(target, prompt_ids):
            cfg = target.cfg
            return diagnose_retained_fork(
                target, prompt_ids, expected_prefix=args.diagnose_retained_prefix,
                kv_type=KVCache, state_digest=_state_digest,
                metadata_digest=lambda kv: complete_state_metadata(
                    kv, array_digest=array_digest),
                copy_ple=lambda value: copy_ple_bits(
                    value, array_module=mx, numpy_module=np),
                synchronize_ple=synchronize_ple,
                pressure=lambda: {**_pressure(),
                                  "rss_bytes": psutil.Process().memory_info().rss},
                metal_memory=lambda: {"active_bytes": int(mx.get_active_memory()),
                                      "cache_bytes": int(mx.get_cache_memory()),
                                      "allocator_peak_bytes": int(mx.get_peak_memory())},
                clear_cache=mx.clear_cache,
                include_kda_conv=args.diagnose_retained_kda_conv,
                expected_kda_layers=tuple(
                    layer for layer, kind in enumerate(cfg.layer_types)
                    if kind == "linear_attention"),
                expected_kda_shape=(
                    1, max(0, cfg.linear_conv_kernel_dim - 1),
                    2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
                    + cfg.linear_num_value_heads * cfg.linear_value_head_dim),
                expected_kda_dtype=mx.bfloat16)

    original = server._engine_generate
    probe = FirstTokenHTTPProbe(
        original, artifact=args.artifact, label=args.label,
        expected_prompt_tokens=args.expected_prompt_tokens,
        engine_type=Qwen4MTPSpeculativeEngine, state_digest=_state_digest,
        hidden_digest=lambda value: _hidden_digest(
            value, array_module=mx, numpy_module=np),
        pressure=_pressure,
        metal_memory=lambda: {"active_bytes": int(mx.get_active_memory()),
                              "allocator_peak_bytes": int(mx.get_peak_memory())},
        profile_identity=active_runtime_profile_fields,
        model_revision=_model_revision,
        retained_diagnostic=retained_diagnostic,
    )
    forwarded = ["runtime.server", "--port", str(args.port)]
    for profile in args.profile:
        forwarded.extend(("--profile", profile))
    for directory in args.profile_dir:
        forwarded.extend(("--profile-dir", directory))
    previous_argv = sys.argv
    server._engine_generate = probe
    sys.argv = forwarded
    try:
        server.main()
    finally:
        server._engine_generate = original
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
