"""Pure CPU/cache gates for content-addressed hybrid tool retrieval."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from runtime.tool_embeddings import (
    ENCODER_DIMENSION, EmbeddingConfig, ToolEmbeddingCache, ToolEmbeddingError,
    build_tool_cache, hybrid_scores)
from runtime.toolcalls import rank_tool_indices, tool_search_capsule


def _unit(axis: int):
    vector = np.zeros(ENCODER_DIMENSION, dtype=np.float32)
    vector[axis] = 1.0
    return vector


def _semantic_encoder(calls):
    def encode(texts, _config):
        calls.append(tuple(texts))
        vectors = []
        for text in texts:
            lowered = text.lower()
            if any(word in lowered for word in (
                    "shell", "launch", "background", "orchestrator")):
                vectors.append(_unit(0))
            elif "calendar" in lowered:
                vectors.append(_unit(1))
            else:
                vectors.append(_unit(2))
        return vectors
    return encode


def _config(tmp_path, *, weight=1.0):
    return EmbeddingConfig(
        model_dir=tmp_path / "unused-model",
        cache_dir=tmp_path / "cache",
        semantic_weight=weight,
        timeout_seconds=5.0,
        min_available_mb=4000,
    )


def test_offline_tool_and_online_query_vectors_are_cached_without_raw_text(tmp_path):
    calls = []
    encoder = _semantic_encoder(calls)
    config = _config(tmp_path)
    capsules = [
        "Tool: command_runner\nPurpose: execute a shell command",
        "Tool: event_create\nPurpose: create a calendar appointment",
    ]
    first = build_tool_cache(
        capsules, config=config, encoder=encoder, verify_model=False)
    second = build_tool_cache(
        capsules, config=config, encoder=encoder, verify_model=False)
    assert first["cache_hits"] == 0 and first["encoded"] == 2
    assert second["cache_hits"] == 2 and second["encoded"] == 0
    assert len(calls) == 1

    query = "launch an isolated background workload SECRET_QUERY_71"
    scores, metadata = hybrid_scores(
        capsules, query, [0.0, 100.0], config=config, encoder=encoder)
    assert scores[0] > scores[1]  # semantic vector corrects misleading lexical score
    assert metadata["tool_embedding_status"] == "hybrid"
    assert metadata["tool_embedding_tool_cache_hits"] == 2
    assert metadata["tool_embedding_query_cache_hit"] == 0
    assert len(calls) == 2

    repeat_scores, repeat_metadata = hybrid_scores(
        capsules, query, [0.0, 100.0], config=config, encoder=encoder)
    assert repeat_scores == scores
    assert repeat_metadata["tool_embedding_query_cache_hit"] == 1
    assert len(calls) == 2  # exact query never reloads the encoder

    cache_bytes = b"".join(
        path.read_bytes() for path in config.cache_dir.rglob("*") if path.is_file())
    assert b"SECRET_QUERY_71" not in cache_bytes
    assert b"execute a shell command" not in cache_bytes


def test_corrupt_vector_fails_back_to_complete_lexical_ranking(tmp_path):
    calls = []
    encoder = _semantic_encoder(calls)
    config = _config(tmp_path)
    capsules = ["shell command tool", "calendar event tool"]
    build_tool_cache(capsules, config=config, encoder=encoder, verify_model=False)
    payload = next(config.cache_dir.rglob("*.npy"))
    damaged = bytearray(payload.read_bytes())
    damaged[-1] ^= 0xFF
    payload.write_bytes(damaged)

    lexical = [9.0, 3.0]
    scores, metadata = hybrid_scores(
        capsules, "launch process", lexical, config=config, encoder=encoder)
    assert scores == lexical
    assert metadata["tool_embedding_status"] == "fallback"
    assert metadata["tool_embedding_fallback"] == "offline_tool_cache_incomplete"
    # Serving never repairs/re-encodes a partial catalog while Qwen is resident.
    assert len(calls) == 1


def test_catalog_identity_and_scores_are_permutation_invariant(tmp_path):
    calls = []
    encoder = _semantic_encoder(calls)
    config = _config(tmp_path, weight=0.7)
    capsules = ["shell execution", "calendar appointment", "document reader"]
    build_tool_cache(capsules, config=config, encoder=encoder, verify_model=False)
    scores, meta = hybrid_scores(
        capsules, "launch command", [3.0, 2.0, 1.0],
        config=config, encoder=encoder)

    order = [2, 0, 1]
    permuted_scores, permuted_meta = hybrid_scores(
        [capsules[i] for i in order], "launch command",
        [[3.0, 2.0, 1.0][i] for i in order], config=config, encoder=encoder)
    assert meta["tool_embedding_catalog_id"] == permuted_meta[
        "tool_embedding_catalog_id"]
    assert [permuted_scores[order.index(i)] for i in range(3)] == scores


def test_capsule_is_bounded_structural_and_excludes_defaults():
    secret = "DO_NOT_CACHE_DEFAULT_998"
    tool = {"type": "function", "function": {
        "name": "workspaceExecuteCommand",
        "description": "Run work in a project.",
        "parameters": {
            "type": "object",
            "properties": {
                "program": {
                    "type": "string", "description": "Executable to invoke",
                    "default": secret,
                },
                "mode": {"type": "string", "enum": ["safe", "fast"]},
            },
            "required": ["program"],
        },
    }}
    capsule = tool_search_capsule(tool, max_chars=1000)
    assert len(capsule) <= 1000
    assert "workspace execute command" in capsule
    assert "shell" in capsule and "terminal" in capsule and "cli" in capsule
    assert "program: string required" in capsule
    assert "safe, fast" in capsule
    assert secret not in capsule

    neutral = tool_search_capsule({"type": "function", "function": {
        "name": "catalog_item", "description": "Update an item.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
        }},
    }})
    assert "query: string" in neutral
    assert "discovery" not in neutral and "retrieval" not in neutral


def test_rank_tool_indices_uses_hybrid_cache_and_keeps_canonical_ties(tmp_path):
    tools = [
        {"type": "function", "function": {
            "name": "orchestrator_launch", "description": "Coordinate workloads.",
            "parameters": {"type": "object"},
        }},
        {"type": "function", "function": {
            "name": "calendar_process", "description": "Manage process meetings.",
            "parameters": {"type": "object"},
        }},
    ]
    calls = []
    encoder = _semantic_encoder(calls)
    config = _config(tmp_path)
    build_tool_cache(
        [tool_search_capsule(tool) for tool in tools],
        config=config, encoder=encoder, verify_model=False)
    messages = [{"role": "user", "content": "start a background job"}]
    env = {
        "VMODEL_TOOL_EMBEDDINGS": "1",
        "VMODEL_TOOL_EMBEDDING_CACHE": str(config.cache_dir),
        "VMODEL_TOOL_EMBEDDING_MODEL": str(config.model_dir),
        "VMODEL_TOOL_EMBEDDING_WEIGHT": "1",
    }
    with patch.dict("os.environ", env), \
         patch("runtime.tool_embeddings.encode_texts_subprocess", encoder):
        ranking, metadata = rank_tool_indices(
            tools, messages, use_embeddings=True, return_metadata=True)
    assert ranking[0] == 0
    assert metadata["tool_retrieval_profile"] == (
        "hybrid-bge-lexical-capability-v1")
    assert metadata["tool_embedding_status"] == "hybrid"


def test_missing_offline_catalog_is_safe_lexical_fallback(tmp_path):
    config = _config(tmp_path)
    lexical = [1.5, 0.5]
    scores, metadata = hybrid_scores(
        ["one", "two"], "novel query", lexical, config=config,
        encoder=lambda *_: (_ for _ in ()).throw(AssertionError("must not encode")))
    assert scores == lexical
    assert metadata["tool_embedding_status"] == "fallback"
    assert metadata["tool_embedding_tool_cache_hits"] == 0
    assert metadata["tool_embedding_tool_cache_misses"] == 2

    with patch.dict("os.environ", {"VMODEL_TOOL_EMBEDDINGS_REQUIRED": "1"}):
        try:
            hybrid_scores(
                ["one", "two"], "novel query", lexical, config=config,
                encoder=lambda *_: [])
        except ToolEmbeddingError as error:
            assert "offline_tool_cache_incomplete" in str(error)
        else:
            raise AssertionError("required hybrid mode silently fell back")


def test_cache_object_permissions_are_private(tmp_path):
    cache = ToolEmbeddingCache(tmp_path / "cache")
    cache.store_many("tool", ["private capsule"], [_unit(0)])
    modes = [path.stat().st_mode & 0o777 for path in cache.root.rglob("*") if path.is_file()]
    assert modes and all(mode == 0o600 for mode in modes)


def test_query_vector_cache_is_bounded(tmp_path):
    cache = ToolEmbeddingCache(tmp_path / "cache")
    for index in range(3):
        cache.store_many(
            "query", [f"private query {index}"], [_unit(index)], max_objects=2)
    query_meta = []
    for path in cache.objects.rglob("*.json"):
        import json
        meta = json.loads(path.read_text())
        if meta["kind"] == "query":
            query_meta.append(meta)
    assert len(query_meta) == 2
