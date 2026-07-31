"""Hot-expert staging gates."""

import json

import mlx.core as mx

from formats.fast_tier import _expert_files, _trunk_files, stage_hot_experts
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


def test_trunk_files_exclude_only_independently_paged_routed_experts():
    manifest = {
        "model.language_model.layers.3.self_attn.q_proj.weight": "q.vt",
        "model.language_model.layers.3.mlp.shared_expert.up_proj.weight":
            "shared.vt",
        _name(7, "up_proj", "weight"): "routed.vt",
        "model.language_model.embed_tokens.weight": "embed.vt",
    }

    assert _trunk_files(manifest) == ["q.vt", "shared.vt"]


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


def test_fast_tier_can_prioritize_complete_trunk_before_hot_experts(
        tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    trunk_name = "model.language_model.layers.3.self_attn.q_proj.weight"
    tensors = {trunk_name: mx.arange(64, dtype=mx.float32).reshape(8, 8)}
    for expert in (7, 9):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = _name(expert, projection, "weight")
            tensors[name] = mx.arange(32, dtype=mx.float32).reshape(4, 8)
    mx.save_safetensors(str(model / "model.safetensors"), tensors)
    (model / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (model / "expert_transitions.json").write_text(json.dumps({
        "3,7,7": 10,
        "3,9,9": 5,
    }))
    vpack = pack_model(model, verify_shards=True)
    manifest = json.loads((vpack / "manifest.json").read_text())
    build_from_vpack(model)
    for path in vpack.glob("*.vt"):
        path.unlink()

    trunk_file = manifest[trunk_name]
    reader_size = (
        8 + len(json.dumps(json.loads(
            (model / "weights.vpack2.index.json").read_text()
        )[trunk_name]["head"]).encode())
        + json.loads((model / "weights.vpack2.index.json").read_text())[
            trunk_name]["len"]
    )
    fast_root = tmp_path / "fast"
    monkeypatch.setattr(
        "formats.fast_tier._safe_cache_root", lambda path: path.resolve())
    report = stage_hot_experts(
        model, fast_root,
        budget_bytes=reader_size + 500,
        include_trunk=True,
    )

    assert report["selected_trunk_files"] == 1
    assert report["selected_trunk_bytes"] == reader_size
    assert (fast_root / model.name / trunk_file).is_file()
    assert report["selected_experts"] < 2
