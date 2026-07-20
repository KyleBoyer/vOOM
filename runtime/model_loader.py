"""WeightStore: lazy, per-tensor access to (possibly sharded) safetensors checkpoints.

`mx.load` on a safetensors file returns *lazy* file-backed arrays; nothing is read
until `mx.eval`. Opening a shard costs ~3 ms, so the store re-opens shards on every
fetch rather than holding evaluated arrays — residency is entirely the caller's
(or the WeightCache's) responsibility. Dropping the returned arrays is eviction.
"""

from __future__ import annotations

from bisect import bisect_left
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from .config import ModelConfig
from .local_config import get_storage_config


_QUANT_MODES = {"affine", "mxfp4", "nvfp4", "mxfp8"}


@dataclass(frozen=True)
class _QuantAux:
    scales: str
    biases: str | None
    bits: int
    group_size: int
    mode: str


@dataclass(frozen=True)
class _CTInt4Aux:
    """F93: a vllm-project/compressed-tensors "pack-quantized" INT4 triplet
    (.weight_packed/.weight_scale/.weight_shape in place of a plain
    .weight). Distinct scheme from this project's own QTensor/mx.quantize
    format above -- different library, different bit layout, dequantized
    eagerly to a dense bf16 array at fetch time (see WeightStore.fetch),
    not lazily matmul'd via mx.quantized_matmul like a QTensor."""

    packed: str
    scale: str
    shape: str


def _quant_params(value) -> tuple[int, int, str] | None:
    """Normalize one standard-MLX quantization descriptor."""
    if not isinstance(value, dict):
        return None
    try:
        bits = int(value["bits"])
        group_size = int(value["group_size"])
    except (KeyError, TypeError, ValueError):
        return None
    mode = str(value.get("mode", "affine"))
    valid = (
        mode == "affine" and group_size in (32, 64, 128)
        and bits in (2, 3, 4, 5, 6, 8)
    ) or (
        (mode, group_size, bits)
        in {("mxfp4", 32, 4), ("mxfp8", 32, 8), ("nvfp4", 16, 4)}
    )
    if not valid:
        return None
    return bits, group_size, mode


def _read_text_retry(path: Path, attempts: int = 4) -> str:
    """F24: metadata reads on externally-hosted models survive transient
    mount drops (same failure class that killed a GLM run at config-read
    time)."""
    import os as _os
    import time as _t

    for i in range(attempts):
        try:
            return path.read_text()
        except OSError:
            if i == attempts - 1:
                raise
            remount = get_storage_config().remount_command_for(path)
            if remount:
                _os.system(remount)
            _t.sleep(5 * (2 ** i))


class WeightStore:
    def __init__(self, model_dir: str | Path, fast_dirs: list[str | Path] | None = None,
                 *, require_vpack_hashes: bool = False,
                 require_raw_weight_hashes: bool = False):
        """fast_dirs: optional overlay directories on faster disks, ordered
        fastest-first (split placement across N drives). Packed tensor files found
        in an earlier tier are read from there instead of the primary store —
        bytes served from a fast tier leave the slow disk's critical path."""
        self.dir = get_storage_config().resolve(model_dir)
        self.fast_dirs = [Path(d).expanduser() for d in (fast_dirs or [])]
        self.fast_tier_bytes = 0
        self.fast_tier_tensors = 0
        self.archive_bytes = 0
        self.config = ModelConfig.from_dir(self.dir)
        raw_config = json.loads(_read_text_retry(self.dir / "config.json"))
        text_config = raw_config.get("text_config", {})
        quantization = (
            raw_config.get("quantization")
            or raw_config.get("quantization_config")
            or (text_config.get("quantization") if isinstance(text_config, dict) else None)
            or (text_config.get("quantization_config")
                if isinstance(text_config, dict) else None)
            or {}
        )
        self.quantization: dict = dict(quantization) if isinstance(quantization, dict) else {}
        self.on_disk_quantized = False
        self.quantization_identity = "none"
        self.quantized_bytes_per_weight = 0.0
        # Physical expert payload estimate for trace/layout simulation. This is
        # deliberately separate from the materialized WeightCache size: Kimi
        # K2.5 reads released INT4+BF16-scales but currently expands to BF16 in
        # memory, while runtime quantize-on-load does the opposite (reads BF16,
        # retains a smaller QTensor).
        self.expert_storage_bytes_per_weight = 2.0

        # Store preference: vpack2 (sequential archive, coalesced reads) over vpack
        # (per-tensor files) over raw safetensors. Both packed forms are bit-exact.
        self.vpack2 = None
        self.vpack = self.dir / "weights.vpack"
        self._vpack_overlay_manifest: dict[str, str] = {}
        if (self.dir / "weights.vpack2.index.json").exists() or (self.dir / "vpack2.CURRENT").exists():
            import sys

            root = str(Path(__file__).resolve().parent.parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            from formats.packed2 import Vpack2Reader

            self.vpack2 = Vpack2Reader(self.dir, require_hashes=require_vpack_hashes)
            overlay_manifest = self.vpack / "manifest.json"
            if overlay_manifest.exists():
                self._vpack_overlay_manifest = json.loads(
                    _read_text_retry(overlay_manifest))
        self.require_vpack_hashes = require_vpack_hashes
        self.packed = self.vpack2 is not None or (self.vpack / "manifest.json").exists()
        if self.vpack2 is not None:
            self.integrity_mode = self.vpack2.integrity_mode
            self.integrity_identity = self.vpack2.integrity_mode
        elif self.packed:
            self.integrity_mode = "legacy-vpack-no-body-hash"
            self.integrity_identity = self.integrity_mode
        elif require_raw_weight_hashes:
            from .weight_integrity import verify_manifest

            digest = verify_manifest(self.dir)
            self.integrity_mode = "raw-safetensors-sha256"
            self.integrity_identity = f"raw-sha256-{digest}"
        else:
            self.integrity_mode = "raw-safetensors-stat"
            self.integrity_identity = self.integrity_mode
        if self.vpack2 is not None:
            self.weight_map = {n: "weights.vpack2" for n in self.vpack2.index}
        elif self.packed:
            self.weight_map = json.loads(_read_text_retry(self.vpack / "manifest.json"))
        else:
            index_path = self.dir / "model.safetensors.index.json"
            if index_path.exists():
                self.weight_map: dict[str, str] = json.loads(_read_text_retry(index_path))["weight_map"]
            else:
                single = self.dir / "model.safetensors"
                self.weight_map = {name: single.name for name in mx.load(str(single))}

        # Qwen3-VL-class checkpoints nest the text model under
        # model.language_model.*: expose canonical model.* aliases so the
        # dense engine runs unchanged. visual.* names pass through untouched
        # (the vision tower addresses them explicitly).
        # F93 (2026-07-19): Kimi K2.5 uses the OPPOSITE order,
        # language_model.model.*, not model.language_model.* -- confirmed
        # against the real downloaded checkpoint's model.safetensors.index.json.
        # Both prefixes canonicalize to the same "model.*" the rest of the
        # engine already expects.
        #
        # 2026-07-19 (later): K2.5's lm_head is NOT nested under
        # language_model.model.* at all -- it's a sibling at
        # language_model.lm_head.weight (the wrapper's own top-level
        # attribute, not inside the inner text-model submodule). The two
        # branches above never matched it, so store.has("lm_head.weight")
        # was silently False for K2.5 -- engine.py's _lm_head_weight()
        # fallback (self.cache.get("lm_head", ["lm_head.weight"])) would
        # have raised a KeyError the first time a real request reached
        # final-logit computation (never yet exercised: every K2.5 test so
        # far failed on a memory-governor rejection during layer streaming,
        # long before logits). Generalized: strip "language_model." for ANY
        # top-level key under it, not just the ".model." submodule case.
        self._real_name: dict[str, str] = {}
        for n in list(self.weight_map):
            if n.startswith("model.language_model."):
                canon = "model." + n[len("model.language_model."):]
            elif n.startswith("language_model.model."):
                canon = "model." + n[len("language_model.model."):]
            elif n.startswith("language_model."):
                canon = n[len("language_model."):]
            else:
                continue
            self._real_name[canon] = n
            self.weight_map[canon] = self.weight_map.pop(n)

        # Standard MLX quantized checkpoints store one logical matrix as
        # ``name.weight`` plus row/group metadata in ``name.scales`` and,
        # for affine quantization, ``name.biases``. Expose only the logical
        # matrix to the scheduler and remember which physical tensors must be
        # fetched together to reconstruct a QTensor.
        self._quant_aux: dict[str, _QuantAux] = {}
        quant_aux_names: set[str] = set()
        if not self.packed:
            global_params = _quant_params(self.quantization)
            for name in list(self.weight_map):
                if not name.endswith(".weight"):
                    continue
                stem = name[:-len(".weight")]
                scales = f"{stem}.scales"
                biases = f"{stem}.biases"
                if scales in self.weight_map:
                    real_stem = self._real_name.get(name, name)[:-len(".weight")]
                    configured = self.quantization.get(
                        stem, self.quantization.get(real_stem, None))
                    params = _quant_params(configured) or global_params
                    if params is None:
                        raise ValueError(
                            f"standard MLX quantized tensor {name!r} has scales but "
                            "no usable bits/group_size descriptor in config.json"
                        )
                    bits, group_size, mode = params
                    bias_name = biases if biases in self.weight_map else None
                    self._quant_aux[name] = _QuantAux(
                        scales, bias_name, bits, group_size, mode)
                    quant_aux_names.add(scales)
                    if bias_name is not None:
                        quant_aux_names.add(bias_name)

        # F93: vllm-project/compressed-tensors "pack-quantized" INT4 --
        # confirmed on Kimi K2.5's real checkpoint, MoE expert weights only
        # (attention/router stay plain bf16 .weight tensors). Expose only
        # the logical "<stem>.weight" name; fetch() dequantizes the
        # packed/scale/shape triplet to a dense array (see _CTInt4Aux).
        self._ct_int4_aux: dict[str, _CTInt4Aux] = {}
        if not self.packed:
            for name in list(self.weight_map):
                if not name.endswith(".weight_packed"):
                    continue
                stem = name[:-len(".weight_packed")]
                scale = f"{stem}.weight_scale"
                shape = f"{stem}.weight_shape"
                if scale not in self.weight_map or shape not in self.weight_map:
                    continue
                logical = f"{stem}.weight"
                self._ct_int4_aux[logical] = _CTInt4Aux(name, scale, shape)
                self.weight_map[logical] = self.weight_map[name]
                quant_aux_names.add(name)
                quant_aux_names.add(scale)
                quant_aux_names.add(shape)

        packed_triplets = any(
            name.endswith(".weight")
            and f"{name[:-len('.weight')]}.scales" in self.weight_map
            for name in self.weight_map
        )
        if self.packed and packed_triplets:
            raise NotImplementedError(
                "packing standard MLX weight/scales/biases triplets is not yet "
                "supported; use the original safetensors checkpoint"
            )

        self.on_disk_quantized = bool(self._quant_aux)
        if self.on_disk_quantized:
            identity = {
                name: {
                    "bits": aux.bits,
                    "group_size": aux.group_size,
                    "mode": aux.mode,
                }
                for name, aux in sorted(self._quant_aux.items())
            }
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:16]
            self.quantization_identity = f"mlx-{digest}"
            self.quantized_bytes_per_weight = max(
                aux.bits / 8 + (
                    8 / aux.group_size if aux.mode == "affine"
                    else 1 / aux.group_size
                )
                for aux in self._quant_aux.values()
            )
            self.expert_storage_bytes_per_weight = (
                self.quantized_bytes_per_weight)
        elif self._ct_int4_aux:
            # compressed-tensors groupwise INT4 stores 4 payload bits/weight
            # plus one BF16 scale per group. Read the released descriptor rather
            # than hard-coding K2.5's currently observed group size 32.
            candidates = []
            for group in self.quantization.get("config_groups", {}).values():
                weights = group.get("weights", {}) if isinstance(group, dict) else {}
                try:
                    bits = int(weights["num_bits"])
                    group_size = int(weights["group_size"])
                except (KeyError, TypeError, ValueError):
                    continue
                if bits > 0 and group_size > 0:
                    candidates.append(bits / 8 + 2 / group_size)
            # An unfamiliar descriptor must not make an otherwise-decodable
            # checkpoint unloadable merely for telemetry; retain the safe BF16
            # fallback when the physical format cannot be priced here.
            if candidates:
                self.expert_storage_bytes_per_weight = max(candidates)
        elif self.config.model_type == "gpt_oss":
            # Released MXFP4 blocks plus scales/metadata. Match the existing
            # conservative resident/admission estimate used by engine.py.
            self.expert_storage_bytes_per_weight = 0.6
        elif (self.quantization and self.config.model_type != "gpt_oss"
                and not (self.quantization.get("quant_method") == "compressed-tensors"
                         and self.quantization.get("format") == "pack-quantized"
                         and self._ct_int4_aux)):
            # F93: vllm-project/compressed-tensors pack-quantized INT4 is now
            # a supported on-disk layout (see _ct_int4_aux above and
            # fetch()'s dequantize_compressed_tensors_int4 call) -- only
            # reaches this branch, and still raises, if the declared
            # quantization method ISN'T that exact one, or claims to be but
            # no matching .weight_packed/.weight_scale/.weight_shape
            # triplets were actually found (config/checkpoint mismatch).
            method = self.quantization.get("quant_method", "unknown")
            standard_declared = (
                _quant_params(self.quantization) is not None
                or any(_quant_params(value) is not None
                       for value in self.quantization.values())
            )
            suspicious_scales = any(
                "scale_inv" in name or name.endswith(".weight_scale")
                for name in self.weight_map
            )
            if standard_declared or method != "unknown" or suspicious_scales:
                raise NotImplementedError(
                    f"unsupported on-disk quantization layout ({method}); convert "
                    "the checkpoint to standard MLX weight/scales/biases triplets"
                )

        self._names = sorted(n for n in self.weight_map if n not in quant_aux_names)

    # ---- name queries -------------------------------------------------

    def layer_param_names(self, layer: int) -> list[str]:
        return self.names_with_prefix(f"model.layers.{layer}.")

    def names_with_prefix(self, prefix: str) -> list[str]:
        # `_names` is immutable and sorted after construction. Start at the
        # lexicographic insertion point, then inspect only the contiguous match
        # range instead of rescanning every tensor for every streamed MoE page.
        start = bisect_left(self._names, prefix)
        end = start
        while end < len(self._names) and self._names[end].startswith(prefix):
            end += 1
        return self._names[start:end]

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def estimate_expert_storage_page_bytes(
            self, expert_prefix: str, fallback: int) -> int:
        """Return an average physical expert page size without reading weights.

        Vpack2 already has exact compressed extent lengths in its immutable
        index, so use them. Raw/legacy stores fall back to architecture/format
        math; individual safetensors do not expose compressed expert extents.
        The planner still charges pages independently, so this is an average
        byte estimate rather than a claim that every compressed page is equal.
        """
        fallback = int(fallback)
        if self.vpack2 is None:
            return fallback
        marker = f".{expert_prefix}."
        page_bytes: dict[tuple[str, int], int] = defaultdict(int)
        for logical_name in self._names:
            if marker not in logical_name:
                continue
            layer_prefix, suffix = logical_name.split(marker, 1)
            if not layer_prefix.startswith("model.layers."):
                continue
            try:
                expert = int(suffix.split(".", 1)[0])
            except ValueError:
                continue
            physical_name = self._real_name.get(logical_name, logical_name)
            entry = self.vpack2.index.get(physical_name)
            if entry is None:
                continue
            page_bytes[(layer_prefix, expert)] += int(entry["len"])
        if not page_bytes:
            return fallback
        return round(sum(page_bytes.values()) / len(page_bytes))

    def is_quantized(self, name: str) -> bool:
        """Whether one logical matrix is stored as an MLX quantized triplet.

        ``on_disk_quantized`` is checkpoint-wide and therefore too coarse for
        selective artifacts such as expert-only OLMoE. Sidecars and placement
        decisions must ask about the tensor they actually consume.
        """
        return name in self._quant_aux

    def quantization_ratio(self, name: str) -> float:
        """Packed bytes divided by BF16 bytes for one logical matrix."""
        aux = self._quant_aux.get(name)
        if aux is None:
            return 1.0
        metadata = 8 if aux.mode == "affine" else 1
        return (aux.bits / 8 + metadata / aux.group_size) / 2

    def uniform_quantization_ratio(self, name_fragment: str) -> float:
        """Return a safe family ratio only when every matrix is packed.

        A partially quantized family returns 1.0. Applying one optimistic ratio
        to both packed experts and a raw router/projection would make the memory
        planner claim residency that the checkpoint cannot provide.
        """
        names = [
            name for name in self._names
            if name_fragment in name and name.endswith(".weight")
        ]
        if not names or any(name not in self._quant_aux for name in names):
            return 1.0
        return max(self.quantization_ratio(name) for name in names)

    # ---- fetching -----------------------------------------------------

    def _fetch_vpack2_fast_overlay(
        self, physical_names: list[str],
    ) -> tuple[dict[str, mx.array], list[str], int]:
        """Read exact vpack tensor copies from faster disks before the archive.

        A vpack2 archive is ideal for the external drive's sequential floor,
        while a small internal tier benefits from independently copied hot
        tensors. The copied `.vt` body is authenticated against vpack2's source
        hash on the same read that decodes it, so tiering cannot weaken the
        released-byte correctness contract.
        """
        if (not self.fast_dirs or not self._vpack_overlay_manifest
                or self.vpack2 is None):
            return {}, list(physical_names), 0
        selected: list[tuple[str, Path]] = []
        remaining: list[str] = []
        for name in physical_names:
            filename = self._vpack_overlay_manifest.get(name)
            path = next(
                (root / filename for root in self.fast_dirs
                 if filename and (root / filename).is_file()),
                None,
            )
            if path is None:
                remaining.append(name)
            else:
                selected.append((name, path))
        if not selected:
            return {}, remaining, 0

        import concurrent.futures as cf
        import struct

        from formats.packed import to_mx
        from formats.packed2 import decode_body

        def read_one(item):
            name, path = item
            payload = path.read_bytes()
            if len(payload) < 8:
                raise IOError(f"truncated fast-tier tensor: {path}")
            header_len = struct.unpack("<Q", payload[:8])[0]
            header_end = 8 + header_len
            if header_end > len(payload):
                raise IOError(f"truncated fast-tier header: {path}")
            try:
                header = json.loads(payload[8:header_end])
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise IOError(f"invalid fast-tier header: {path}") from error
            entry = self.vpack2.index[name]
            if header != entry["head"]:
                raise IOError(f"fast-tier tensor metadata mismatch: {name}")
            body = self.vpack2._checked_body(name, entry, payload[header_end:])
            if len(body) != int(entry["len"]):
                raise IOError(f"fast-tier tensor extent mismatch: {name}")
            return name, header, decode_body(header, body), len(body)

        workers = min(4, len(selected))
        if workers == 1:
            decoded = [read_one(selected[0])]
        else:
            with cf.ThreadPoolExecutor(max_workers=workers) as pool:
                decoded = list(pool.map(read_one, selected))
        output = {
            name: to_mx(header, raw)
            for name, header, raw, _length in decoded
        }
        mx.eval(list(output.values()))
        return output, remaining, sum(value[3] for value in decoded)

    def fetch(self, names: list[str]) -> tuple[dict[str, mx.array], float, int]:
        """Materialize tensors; return arrays, wall seconds, store-accounted bytes.

        For raw safetensors the byte field is requested logical tensor payload,
        not an OS/device measurement of physical reads or SMB traffic. Packed
        backends may account compressed extents. Callers must not label this
        field "physical bytes" without independent process/device counters.
        """
        if self.vpack2 is not None or self.packed:
            # Packed reads perform real I/O and decode inside the call. Retry the
            # whole transaction just like raw safetensors, reopening vpack2 after
            # a remount so a cycled mountpoint (e.g. Plex -> Plex-N) cannot remain stale.
            t0 = time.perf_counter()
            for attempt in range(4):
                try:
                    if self.vpack2 is not None:
                        # The archive index retains released physical names,
                        # while multimodal wrappers are exposed to the engine
                        # through canonical model.* aliases. Translate at the
                        # archive boundary and translate results back; passing
                        # canonical Qwen3-VL/Qwen3.6/K2.5 names directly into a
                        # physical-name index otherwise raises a late KeyError.
                        physical = [self._real_name.get(name, name)
                                    for name in names]
                        fetched, remaining, fast_bytes = (
                            self._fetch_vpack2_fast_overlay(physical))
                        nbytes = fast_bytes
                        self.fast_tier_bytes += fast_bytes
                        self.fast_tier_tensors += len(fetched)
                        if remaining:
                            archive, _, archive_bytes = self.vpack2.fetch(remaining)
                            fetched.update(archive)
                            nbytes += archive_bytes
                            self.archive_bytes += archive_bytes
                        out = {
                            logical: fetched[stored]
                            for logical, stored in zip(names, physical)
                        }
                    else:
                        out, _, nbytes = self._fetch_packed(names)
                    return out, time.perf_counter() - t0, nbytes
                except (OSError, RuntimeError, EOFError):
                    mx.clear_cache()
                    if attempt == 3:
                        raise
                    self._recover_nas_mount()
                    if self.vpack2 is not None:
                        from formats.packed2 import Vpack2Reader

                        self.vpack2 = Vpack2Reader(
                            self.dir, require_hashes=self.require_vpack_hashes
                        )
                    time.sleep(5 * (2 ** attempt))
        physical_names: list[str] = []
        seen: set[str] = set()
        for n in names:
            ct_aux = self._ct_int4_aux.get(n)
            if ct_aux is not None:
                # F93: the logical "<stem>.weight" name has NO physical
                # tensor of its own here (unlike _quant_aux, where the
                # logical name IS a real quantized weight tensor plus
                # sidecars) -- only packed/scale/shape physically exist.
                expanded = (ct_aux.packed, ct_aux.scale, ct_aux.shape)
            else:
                aux = self._quant_aux.get(n)
                expanded = ((n, aux.scales, aux.biases) if aux is not None else (n,))
            for physical in expanded:
                if physical is not None and physical not in seen:
                    physical_names.append(physical)
                    seen.add(physical)

        by_shard: dict[str, list[str]] = defaultdict(list)
        for n in physical_names:
            by_shard[self.weight_map[n]].append(n)

        # mx.load() only creates lazy file-backed arrays. The SMB read that can
        # fail happens in mx.eval(), so retry the complete load+select+eval
        # transaction, not just the cheap metadata/open operation.
        t0 = time.perf_counter()
        for attempt in range(4):
            out: dict[str, mx.array] = {}
            try:
                for shard, shard_names in by_shard.items():
                    lazy = self._load_shard(self.dir / shard)
                    for n in shard_names:
                        out[n] = lazy[self._real_name.get(n, n)]
                mx.eval(list(out.values()))
                nbytes = sum(a.nbytes for a in out.values())
                if self._quant_aux or self._ct_int4_aux:
                    from .quant import QTensor, dequantize_compressed_tensors_int4

                    logical: dict = {}
                    for name in names:
                        ct_aux = self._ct_int4_aux.get(name)
                        if ct_aux is not None:
                            shape = tuple(int(v) for v in out[ct_aux.shape].tolist())
                            dequant = dequantize_compressed_tensors_int4(
                                out[ct_aux.packed], out[ct_aux.scale], shape)
                            # 2026-07-19: MLX >=0.31.2 binds ops to their
                            # creation thread's stream (streams are now
                            # thread-local). dequantize_compressed_tensors_int4
                            # returns a LAZY graph (arange/shift/mask/reshape/
                            # cast, never eval'd internally) -- unlike the raw
                            # `out` tensors just above, which were eval'd on
                            # THIS thread at line 428. Left lazy, this graph is
                            # first materialized wherever the caller (the main
                            # thread, via matmul) eventually evals it -- but
                            # when this fetch runs on the prefetch thread
                            # (K2.5's compressed-tensors INT4 experts are the
                            # only checkpoint format that takes this branch),
                            # that is a DIFFERENT thread than the one the ops
                            # were constructed on, and its stream is not
                            # registered there: "RuntimeError: There is no
                            # Stream(gpu, N) in current thread." Force eval
                            # here, on the thread that built the graph.
                            mx.eval(dequant)
                            logical[name] = dequant
                            continue
                        aux = self._quant_aux.get(name)
                        if aux is None:
                            logical[name] = out[name]
                            continue
                        logical[name] = QTensor(
                            out[name], out[aux.scales],
                            out[aux.biases] if aux.biases is not None else None,
                            aux.bits, aux.group_size, aux.mode,
                        )
                    out = logical
                return out, time.perf_counter() - t0, nbytes
            except (OSError, RuntimeError):
                # Discard every partially materialized/lazy array before retry;
                # otherwise stale file descriptors and half-read allocations can
                # survive into the next attempt.
                out.clear()
                mx.clear_cache()
                if attempt == 3:
                    raise
                self._recover_nas_mount()
                time.sleep(5 * (2 ** attempt))

        raise AssertionError("unreachable raw fetch retry state")

    def _recover_nas_mount(self) -> None:
        """Remount/re-resolve this model after a transient storage failure."""
        storage = get_storage_config()
        if not storage.is_configured_path(self.dir):
            return
        candidate = storage.resolve(self.dir)
        if candidate != self.dir:
            print(f"[store] re-resolved model dir -> {candidate}", flush=True)
            self.dir = candidate
            self.vpack = candidate / "weights.vpack"

    def _fetch_packed(self, names: list[str]) -> tuple[dict[str, mx.array], float, int]:
        import sys

        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        from formats.packed import read_tensor_bytes, to_mx

        t0 = time.perf_counter()
        out: dict = {}
        nbytes = 0
        eval_list = []
        by_file: dict[str, list[str]] = defaultdict(list)
        for n in names:
            by_file[self.weight_map[n]].append(n)
        for fname, fnames in by_file.items():
            root = next((d for d in self.fast_dirs if (d / fname).exists()), self.vpack)
            if fname.endswith(".safetensors"):
                # SQ q4 stores: pre-quantized expert triplets (-> QTensor) and
                # bf16 remainder chunks live in plain safetensors files.
                lazy = self._load_shard(root / fname)
                nbytes += (root / fname).stat().st_size
                for n in fnames:
                    if f"{n}.wq" in lazy:
                        from .quant import QTensor

                        q = QTensor(lazy[f"{n}.wq"], lazy[f"{n}.scales"],
                                    lazy[f"{n}.biases"], 4, 64)
                        out[n] = q
                        eval_list += [q.wq, q.scales, q.biases]
                    else:
                        out[n] = lazy[n]
                        eval_list.append(lazy[n])
                continue
            for n in fnames:
                nbytes += (root / self.weight_map[n]).stat().st_size
                head, raw = read_tensor_bytes(root, self.weight_map[n])
                out[n] = to_mx(head, raw)
                eval_list.append(out[n])
        mx.eval(eval_list)
        return out, time.perf_counter() - t0, nbytes

    def _load_shard(self, path: Path, attempts: int = 4):
        """Tier-3 (network storage) resilience: SMB shares drop mid-run — a
        44-minute GLM sweep died to exactly this. Retry with backoff; if the
        volume itself vanished, attempt to remount the configured share."""
        import os
        import time as _t

        for i in range(attempts):
            try:
                return mx.load(str(path))
            except RuntimeError:
                if i == attempts - 1:
                    raise
                if not path.exists() and get_storage_config().is_configured_path(path):
                    self._recover_nas_mount()
                    _t.sleep(3)
                    path = self.dir / path.name
                _t.sleep(5 * (2 ** i))

    def fetch_layer(self, layer: int) -> tuple[dict[str, mx.array], float, int]:
        return self.fetch(self.layer_param_names(layer))
