import json
import struct
from pathlib import Path

import pytest

from runtime.qwen_mtp_quant_clone import (
    EXPECTED_MTP_SPECS,
    FAST_ALIAS_MANIFEST_NAME,
    FAST_ALIAS_SCHEMA,
    FAST_MANIFEST_NAME,
    INDEX_NAME,
    MANIFEST_NAME,
    MANIFEST_SCHEMA,
    build,
    build_fast_alias,
    plan,
    plan_fast_alias,
)


def _write_sparse_safetensors(path: Path, specs=None) -> None:
    specs = dict(EXPECTED_MTP_SPECS if specs is None else specs)
    offset = 0
    header = {}
    for name, (dtype, shape, nbytes) in sorted(specs.items()):
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    payload = json.dumps(header, separators=(",", ":")).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)
        # Sparse truncate proves the declared extents fit without writing the
        # real 225.7 MB proposal payload during this metadata-only unit test.
        handle.truncate(8 + len(payload) + offset)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    shard = "model-00018-of-00018.safetensors"
    _write_sparse_safetensors(source / shard)
    (source / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"},
        "voom_quantization": {
            "profile": "all",
            "source": str(tmp_path / "released-source"),
        },
    }))
    (source / "tokenizer.json").write_text("{}")
    (source / "mtp-bf16.safetensors").write_bytes(b"sidecar")
    (source / "mtp-bf16.manifest.json").write_text("{}")
    (source / INDEX_NAME).write_text(json.dumps({
        "metadata": {
            "total_size": 1,
            "mtplx_mtp_sidecar": "mtp-bf16.safetensors",
            "mtplx_mtp_sidecar_manifest": "mtp-bf16.manifest.json",
            "mtplx_mtp_sidecar_sha256": "0" * 64,
        },
        "weight_map": {"model.embed_tokens.weight": shard},
    }))
    return source


def test_plan_requires_exact_qwen38_packed_mtp_schema(tmp_path):
    source = _source(tmp_path)
    record = plan(source, tmp_path / "clone")
    assert record["schema"] == MANIFEST_SCHEMA
    assert record["mtp_physical_tensors"] == 23
    assert record["mtp_logical_tensors"] == 15
    assert record["mtp_quantized_matrices"] == 8
    assert record["mtp_payload_bytes"] == 225_659_904

    specs = dict(EXPECTED_MTP_SPECS)
    specs["mtp.fc.weight"] = ("U32", (5120, 1279), 26_214_400)
    _write_sparse_safetensors(
        source / "model-00018-of-00018.safetensors", specs)
    with pytest.raises(ValueError, match="schema mismatch"):
        plan(source, tmp_path / "bad-clone")


def test_plan_rejects_truncated_declared_extent(tmp_path):
    source = _source(tmp_path)
    shard = source / "model-00018-of-00018.safetensors"
    with shard.open("r+b") as handle:
        handle.truncate(shard.stat().st_size - 1)
    with pytest.raises(ValueError, match="extent exceeds"):
        plan(source, tmp_path / "clone")


def test_build_is_zero_copy_atomic_and_removes_bf16_sidecar(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "clone"
    record = build(source, output)

    assert output.is_dir()
    assert not (output / "mtp-bf16.safetensors").exists()
    assert not (output / "mtp-bf16.manifest.json").exists()
    assert (output / "tokenizer.json").is_symlink()
    assert (output / "model-00018-of-00018.safetensors").is_symlink()
    index = json.loads((output / INDEX_NAME).read_text())
    assert index["weight_map"]["mtp.fc.weight"] == (
        "model-00018-of-00018.safetensors")
    assert not any(key.startswith("mtplx_mtp_") for key in index["metadata"])
    assert index["metadata"]["vmodel_mtp_proposal_representation"] == (
        "mxfp4-q4-g32")
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert manifest["output_index_sha256"] == record["output_index_sha256"]
    assert manifest["enabled_by_default"] is False
    assert manifest["target_authoritative_required"] is True
    # The clone itself contains only metadata and symlinks, not a copied head.
    regular_bytes = sum(
        path.lstat().st_size for path in output.iterdir() if not path.is_symlink())
    assert regular_bytes < 1_000_000

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build(source, output)


def test_plan_refuses_nested_output_and_non_sidecar_source(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="separate sibling"):
        plan(source, source / "clone")
    index = json.loads((source / INDEX_NAME).read_text())
    index["metadata"].pop("mtplx_mtp_sidecar")
    (source / INDEX_NAME).write_text(json.dumps(index))
    with pytest.raises(ValueError, match="BF16-sidecar"):
        plan(source, tmp_path / "clone")


def test_plan_requires_lossy_converter_provenance(tmp_path):
    source = _source(tmp_path)
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("voom_quantization")
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="lossy provenance"):
        plan(source, tmp_path / "clone")


def test_clone_is_lossy_and_never_generic_default(tmp_path):
    from runtime.server import (
        _is_voom_lossy_checkpoint,
        _preferred_fast_artifact,
    )

    source = _source(tmp_path)
    clone = tmp_path / "clone"
    build(source, clone)
    assert _is_voom_lossy_checkpoint(clone)

    released = tmp_path / "Released"
    released.mkdir()
    (released / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
    }))

    def derivative(name: str, *, disabled: bool) -> Path:
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5",
            "quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"},
            "voom_quantization": {
                "profile": "all", "source": str(released),
            },
        }))
        (path / INDEX_NAME).write_text(json.dumps({"weight_map": {}}))
        if disabled:
            (path / MANIFEST_NAME).write_text(json.dumps({
                "schema": MANIFEST_SCHEMA,
                "enabled_by_default": False,
            }))
        return path

    disabled = derivative("Released-mlx-000-mtpquant", disabled=True)
    ordinary = derivative("Released-mlx-all-mxfp4", disabled=False)
    assert disabled.name < ordinary.name
    assert _preferred_fast_artifact(released) == ordinary


def _fast_source(tmp_path: Path) -> Path:
    source = tmp_path / "fast-root" / "source"
    source.mkdir(parents=True)
    names = (
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.mlp.down_proj.weight",
    )
    manifest = {}
    offset = 0
    for name in names:
        dtype, shape, nbytes = EXPECTED_MTP_SPECS[name]
        manifest[name] = {
            "file": "000016.safetensors",
            "offset": offset,
            "nbytes": nbytes,
            "dtype": dtype,
            "shape": list(shape),
        }
        offset += nbytes
    manifest["model.layers.0.input_layernorm.weight"] = {
        "file": "000016.safetensors",
        "offset": offset,
        "nbytes": 16,
        "dtype": "U8",
        "shape": [16],
    }
    with (source / "000016.safetensors").open("wb") as handle:
        handle.truncate(offset + 16)
    (source / FAST_MANIFEST_NAME).write_text(json.dumps(manifest))
    # Must not leak into the alias: it belongs only to the BF16-sidecar target.
    (source / "mtp-bf16-fast.manifest.json").write_text("{}")
    return source


def test_fast_alias_reuses_two_device_overlay_without_bf16_manifest(tmp_path):
    target_source = _source(tmp_path)
    target = tmp_path / "clone"
    build(target_source, target)
    source = _fast_source(tmp_path)
    output = tmp_path / "fast-clone"
    planned = plan_fast_alias(source, output, target)
    assert planned["schema"] == FAST_ALIAS_SCHEMA
    assert len(planned["mtp_tensors"]) == 5
    built = build_fast_alias(source, output, target)
    assert built == planned
    assert (output / "000016.safetensors").is_symlink()
    assert (output / FAST_MANIFEST_NAME).is_file()
    assert not (output / "mtp-bf16-fast.manifest.json").exists()
    assert json.loads((output / FAST_ALIAS_MANIFEST_NAME).read_text()) == planned


def test_fast_alias_rejects_mismatched_mtp_metadata(tmp_path):
    target_source = _source(tmp_path)
    target = tmp_path / "clone"
    build(target_source, target)
    source = _fast_source(tmp_path)
    manifest = json.loads((source / FAST_MANIFEST_NAME).read_text())
    manifest["mtp.layers.0.mlp.gate_proj.weight"]["shape"] = [1]
    (source / FAST_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="MTP schema mismatch"):
        plan_fast_alias(source, tmp_path / "fast-clone", target)
