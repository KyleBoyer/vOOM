"""Hot-expert staging gates."""

import json

import mlx.core as mx

from formats.fast_tier import _expert_files, stage_hot_experts
from formats.packed import pack_model, read_tensor_bytes
from formats.packed2 import build_from_vpack


def _name(expert: int, projection: str, suffix: str) -> str:
    return (
        f"model.language_model.layers.3.mlp.experts.{expert}."
        f"{projection}.{suffix}")


def test_expert_files_accepts_complete_bf16_and_mlx_quantized_pages():
    manifest = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        bf16 = _name(7, projection, "weight")
        manifest[bf16] = bf16 + ".vt"
        for suffix in ("weight", "scales"):
            quantized = _name(9, projection, suffix)
            manifest[quantized] = quantized + ".vt"

    groups = _expert_files(manifest)

    assert len(groups[(3, 7)]) == 3
    assert len(groups[(3, 9)]) == 6


def test_expert_files_rejects_incomplete_quantized_sidecars():
    manifest = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        weight = _name(11, projection, "weight")
        manifest[weight] = weight + ".vt"
    scale = _name(11, "gate_proj", "scales")
    manifest[scale] = scale + ".vt"

    assert (3, 11) not in _expert_files(manifest)


def test_fast_tier_can_reconstruct_deleted_vpack_files_from_hashed_vpack2(
        tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    tensors = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        name = _name(7, projection, "weight")
        tensors[name] = mx.arange(32, dtype=mx.float32).reshape(4, 8)
    mx.save_safetensors(str(model / "model.safetensors"), tensors)
    (model / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (model / "expert_transitions.json").write_text(
        json.dumps({"3,7,7": 5}))
    vpack = pack_model(model, verify_shards=True)
    manifest = json.loads((vpack / "manifest.json").read_text())
    build_from_vpack(model)
    for path in vpack.glob("*.vt"):
        path.unlink()

    fast_root = tmp_path / "fast"
    monkeypatch.setattr(
        "formats.fast_tier._safe_cache_root", lambda path: path.resolve())
    report = stage_hot_experts(model, fast_root, budget_bytes=1_000_000)

    assert report["selected_experts"] == 1
    assert report["copied_files"] == 3
    for name, filename in manifest.items():
        head, raw = read_tensor_bytes(fast_root / model.name, filename)
        assert head["shape"] == list(tensors[name].shape)
        assert len(raw) == tensors[name].nbytes
