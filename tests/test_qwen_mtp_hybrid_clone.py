import hashlib
import json
import struct
from pathlib import Path

import runtime.qwen_mtp_hybrid_clone as hybrid


def _write_sparse(path: Path, specs: dict[str, tuple[str, tuple[int, ...], int]]):
    offset = 0
    header = {}
    for name, (dtype, shape, size) in sorted(specs.items()):
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    payload = json.dumps(header, separators=(",", ":")).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)
        handle.truncate(8 + len(payload) + offset)


def _fixture(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    bf16_specs = {
        "mtp.fc.weight": ((4, 8), 64),
        "mtp.layers.0.input_layernorm.weight": ((4,), 8),
        "mtp.layers.0.mlp.down_proj.weight": ((4, 8), 64),
        "mtp.layers.0.mlp.gate_proj.weight": ((8, 4), 64),
        "mtp.layers.0.mlp.up_proj.weight": ((8, 4), 64),
        "mtp.layers.0.post_attention_layernorm.weight": ((4,), 8),
        "mtp.layers.0.self_attn.k_norm.weight": ((2,), 4),
        "mtp.layers.0.self_attn.k_proj.weight": ((2, 4), 16),
        "mtp.layers.0.self_attn.o_proj.weight": ((4, 4), 32),
        "mtp.layers.0.self_attn.q_norm.weight": ((2,), 4),
        "mtp.layers.0.self_attn.q_proj.weight": ((4, 4), 32),
        "mtp.layers.0.self_attn.v_proj.weight": ((2, 4), 16),
        "mtp.norm.weight": ((4,), 8),
        "mtp.pre_fc_norm_embedding.weight": ((4,), 8),
        "mtp.pre_fc_norm_hidden.weight": ((4,), 8),
    }
    monkeypatch.setattr(hybrid, "EXPECTED_BF16_SPECS", bf16_specs)
    sidecar = source / "mtp-bf16.safetensors"
    _write_sparse(sidecar, {
        name: ("BF16", shape, size)
        for name, (shape, size) in bf16_specs.items()
    })
    packed_specs = {}
    for name in hybrid.MLP_MATRICES:
        packed_specs[name] = ("U32", (4, 1), 16)
        packed_specs[name.removesuffix(".weight") + ".scales"] = (
            "U8", (4, 1), 4)
    shard = source / "model-00018-of-00018.safetensors"
    _write_sparse(shard, packed_specs)
    (source / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"},
    }))
    (source / "tokenizer.json").write_text("{}")
    sidecar_sha = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    (source / hybrid.INDEX_NAME).write_text(json.dumps({
        "metadata": {
            "mtplx_mtp_sidecar": sidecar.name,
            "mtplx_mtp_sidecar_sha256": sidecar_sha,
        },
        "weight_map": {"model.embed_tokens.weight": shard.name},
    }))
    return source


def test_hybrid_clone_is_zero_copy_and_mlp_only(tmp_path, monkeypatch):
    source = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "hybrid"
    record = hybrid.build(source, output)

    assert record["schema"] == hybrid.HYBRID_MANIFEST_SCHEMA
    assert record["mtp_payload_bytes"] < record["mtp_released_bf16_bytes"]
    assert set(record["mtp_quantized_matrices"]) == hybrid.MLP_MATRICES
    assert (output / "mtp-bf16.safetensors").is_symlink()
    index = json.loads((output / hybrid.INDEX_NAME).read_text())
    metadata = index["metadata"]
    assert metadata["vmodel_mtp_proposal_representation"] == (
        hybrid.REPRESENTATION)
    assert not any(name.startswith("mtplx_mtp_") for name in metadata)
    for name in hybrid.MLP_MATRICES:
        assert index["weight_map"][name] == "model-00018-of-00018.safetensors"
        assert index["weight_map"][
            name.removesuffix(".weight") + ".scales"
        ] == "model-00018-of-00018.safetensors"
    for name in metadata["vmodel_mtp_proposal_plain_names"]:
        assert index["weight_map"][name] == "mtp-bf16.safetensors"


def test_hybrid_clone_rejects_sidecar_hash_mismatch(tmp_path, monkeypatch):
    source = _fixture(tmp_path, monkeypatch)
    index_path = source / hybrid.INDEX_NAME
    index = json.loads(index_path.read_text())
    index["metadata"]["mtplx_mtp_sidecar_sha256"] = "0" * 64
    index_path.write_text(json.dumps(index))
    try:
        hybrid.plan(source, tmp_path / "hybrid")
    except ValueError as error:
        assert "SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("hybrid plan accepted a mismatched sidecar hash")
