"""Local end-to-end gates for dense resident/pipelined lossy decode."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_dense_fixture(path: Path, *, heads: int = 4,
                         kv_heads: int = 2) -> None:
    from tests.fixtures.build_glm_fixture import write_tokenizer

    hidden, intermediate, vocab = 64, 128, 512
    head_dim = hidden // heads
    counter = [0]

    def weight(*shape):
        counter[0] += 1
        mx.random.seed(7000 + counter[0])
        return (mx.random.normal(shape) * 0.05).astype(mx.bfloat16)

    tensors = {
        "model.embed_tokens.weight": weight(vocab, hidden),
        "model.norm.weight": weight(hidden),
    }
    for layer in range(2):
        p = f"model.layers.{layer}"
        tensors.update({
            f"{p}.input_layernorm.weight": weight(hidden),
            f"{p}.self_attn.q_proj.weight": weight(heads * head_dim, hidden),
            f"{p}.self_attn.q_proj.bias": weight(heads * head_dim),
            f"{p}.self_attn.k_proj.weight": weight(kv_heads * head_dim, hidden),
            f"{p}.self_attn.k_proj.bias": weight(kv_heads * head_dim),
            f"{p}.self_attn.v_proj.weight": weight(kv_heads * head_dim, hidden),
            f"{p}.self_attn.v_proj.bias": weight(kv_heads * head_dim),
            f"{p}.self_attn.o_proj.weight": weight(hidden, hidden),
            f"{p}.post_attention_layernorm.weight": weight(hidden),
            f"{p}.mlp.gate_proj.weight": weight(intermediate, hidden),
            f"{p}.mlp.up_proj.weight": weight(intermediate, hidden),
            f"{p}.mlp.down_proj.weight": weight(hidden, intermediate),
        })
    mx.eval(tensors)
    mx.save_safetensors(str(path / "model.safetensors"), tensors)
    (path / "config.json").write_text(json.dumps({
        "model_type": "qwen2",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": 2,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "vocab_size": vocab,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "max_position_embeddings": 512,
        "tie_word_embeddings": True,
        "attention_bias": True,
        "head_dim": head_dim,
        "eos_token_id": [2],
        "torch_dtype": "bfloat16",
    }))
    write_tokenizer(path, vocab)


def _generate(path: Path, fast: bool, max_tokens: int, eos=(), stop=None,
              fused_swiglu: bool = False, last_token_separate: bool = False):
    from runtime.engine import RuntimeConfig, StreamingEngine

    engine = StreamingEngine(str(path), RuntimeConfig(
        max_weight_cache_mb=100,
        quant_bits=4,
        quant_group_size=32,
        quant_mode="mxfp4",
        quant_min_dim=0,
        quantize_tied_lm_head=True,
        resident_fast_decode=fast,
        fused_swiglu=fused_swiglu,
        prefill_chunk_size=2048 if last_token_separate else 0,
        prefill_last_token_separate=last_token_separate,
        governor=False,
    ))
    try:
        engine.cfg.eos_token_ids = tuple(eos)
        result = engine.generate(
            "dense pipeline proof", max_tokens=max_tokens, stop=stop)
        return result, engine.last_kv.offset
    finally:
        engine.close()


def test_pipelined_resident_decode_matches_synchronized_tokens_and_kv(tmp_path):
    _build_dense_fixture(tmp_path)
    slow, slow_offset = _generate(tmp_path, False, 12)
    fast, fast_offset = _generate(tmp_path, True, 12)

    assert fast["tokens"] == slow["tokens"]
    assert fast_offset == slow_offset == fast["prompt_tokens"] + len(fast["tokens"]) - 1
    assert slow["path_stats"]["resident_pipelined_decode_steps"] == 0
    assert fast["path_stats"]["resident_pipelined_decode_steps"] == 11
    assert fast["path_stats"]["resident_fast_decode_sweeps"] == 11


def test_bounded_resident_prefill_matches_synchronized_tokens(tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine

    _build_dense_fixture(tmp_path)
    engine = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100,
        resident_fast_decode=True,
        governor=False,
    ))
    try:
        # The first ordinary request admits every exact BF16 layer. Resident
        # prefill is an optimization of subsequent requests, never a preload.
        engine.generate("warm resident layers", max_tokens=2)

        engine.rc.resident_fast_prefill_limit = 512
        fast = engine.generate("bounded resident prefill proof", max_tokens=12)
        engine.rc.resident_fast_prefill_limit = 0
        baseline = engine.generate("bounded resident prefill proof", max_tokens=12)

        assert fast["tokens"] == baseline["tokens"]
        assert fast["path_stats"]["resident_fast_prefill_sweeps"] == 1
        assert baseline["path_stats"]["resident_fast_prefill_sweeps"] == 0
    finally:
        engine.close()


def test_position_free_engine_matches_tokens_and_releases_pool(tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.kv_cache import PositionFreeKVCache

    # head_dim=32 exercises the real Metal page-table kernel geometry.
    _build_dense_fixture(tmp_path, heads=2, kv_heads=1)
    common = dict(
        max_weight_cache_mb=100,
        resident_fast_decode=True,
        prefill_chunk_size=16,
        hot_prompt_kv=True,
        hot_prompt_kv_chunk_size=16,
        hot_prompt_kv_min_tokens=0,
        tool_pic=True,
        governor=False,
    )
    baseline = StreamingEngine(str(tmp_path), RuntimeConfig(**common))
    shared = StreamingEngine(str(tmp_path), RuntimeConfig(
        **common, tool_pic_shared_pages=True))
    try:
        prompt = "position free engine lifecycle proof"
        expected = baseline.generate(prompt, max_tokens=12)
        candidate = shared.generate(prompt, max_tokens=12)
        assert candidate["tokens"] == expected["tokens"]
        assert candidate["path_stats"]["kv_layout"] == "position_free_shared"
        assert isinstance(shared.last_kv, PositionFreeKVCache)
        assert shared.last_kv.rotated_view_nbytes() == 0
        assert shared._position_free_pool.live_pages == shared.last_kv.offset

        pool = shared._position_free_pool
        shared.release_request_state()
        assert pool.live_pages == 0
    finally:
        baseline.close()
        shared.close()


def test_position_free_engine_rejects_durable_or_unscoped_configuration():
    from runtime.engine import RuntimeConfig, StreamingEngine

    with pytest.raises(ValueError, match="requires tool_pic"):
        StreamingEngine("unused", RuntimeConfig(tool_pic_shared_pages=True))
    with pytest.raises(ValueError, match="engine-local"):
        StreamingEngine("unused", RuntimeConfig(
            tool_pic=True,
            tool_pic_shared_pages=True,
            hot_prompt_kv=True,
            prefill_chunk_size=16,
            hot_prompt_kv_chunk_size=16,
            prompt_kv_dir="durable-is-not-supported",
        ))


def test_position_free_engine_pic_consumes_source_without_leaking_pages(tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine

    _build_dense_fixture(tmp_path, heads=2, kv_heads=1)
    engine = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100,
        prefill_chunk_size=4,
        hot_prompt_kv=True,
        hot_prompt_kv_chunk_size=4,
        hot_prompt_kv_min_tokens=0,
        tool_pic=True,
        tool_pic_shared_pages=True,
        tool_pic_repair_tokens=1,
        tool_pic_min_savings=1,
        governor=False,
    ))
    try:
        source_tokens = list(range(20, 40))
        source = SimpleNamespace(
            token_ids=source_tokens,
            tool_capsules=(("a", 4, 10), ("b", 10, 16)),
        )
        first = engine.generate(source, max_tokens=1)
        assert first["path_stats"]["tool_pic"] == 0
        assert len(engine._hot_prompt_slots) == 1
        source_ids = engine._hot_prompt_slots[0].kv.page_ids

        edited_tokens = source_tokens[:10] + [90, 91] + source_tokens[10:]
        edited = SimpleNamespace(
            token_ids=edited_tokens,
            tool_capsules=(("a", 4, 10), ("new", 10, 12), ("b", 12, 18)),
        )
        result = engine.generate(edited, max_tokens=1)
        assert result["path_stats"]["tool_pic"] == 1
        assert len(engine._hot_prompt_slots) == 1
        destination = engine._hot_prompt_slots[0].kv
        assert destination.offset == len(edited_tokens)
        # At least one unchanged tool tail still points at the original physical
        # ids, while source-only pages were released after ownership transfer.
        assert set(source_ids) & set(destination.page_ids)
        assert engine._position_free_pool.live_pages == destination.offset
    finally:
        engine.close()


def test_explicit_raw_prefetch_worker_count_is_honored(tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine

    _build_dense_fixture(tmp_path)
    engine = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100,
        prefetch_depth=1,
        prefetch_workers=2,
        governor=False,
    ))
    try:
        assert len(engine.prefetcher._workers) == 2
        assert engine.generate("prefetch workers", max_tokens=2)["tokens"]
    finally:
        engine.close()


def test_serial_position_verifier_matches_one_token_target_sweeps(tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine

    _build_dense_fixture(tmp_path)
    engine = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, governor=False))
    try:
        prefix = engine.tokenizer.encode("serial verifier prefix").ids
        window = engine.tokenizer.encode("candidate window proof").ids[:4]
        sequential_kv = engine.new_kv()
        serial_kv = engine.new_kv()
        engine.forward_tokens(prefix, sequential_kv)
        engine.forward_tokens(prefix, serial_kv)

        sequential = []
        for token in window:
            sequential.append(engine.forward_tokens([token], sequential_kv)[-1])
        sequential = mx.stack(sequential)
        serial = engine.forward_tokens_serial_positions(window, serial_kv)
        mx.eval(sequential, serial)

        assert mx.array_equal(serial, sequential)
        for left, right in zip(serial_kv.keys, sequential_kv.keys):
            assert mx.array_equal(left, right)
        for left, right in zip(serial_kv.values, sequential_kv.values):
            assert mx.array_equal(left, right)
    finally:
        engine.close()


def test_dense_speculation_matches_target_only_with_serial_verifier(tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.speculative import SpeculativeDecoder

    _build_dense_fixture(tmp_path)
    target = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, governor=False))
    draft = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, governor=False))
    try:
        baseline = target.generate("dense speculative proof", max_tokens=12)
        speculative = SpeculativeDecoder(target, draft, k=3).generate(
            "dense speculative proof", max_tokens=12)

        assert speculative["tokens"] == baseline["tokens"]
        assert speculative["stats"].sweeps < len(baseline["tokens"])
        assert speculative["prompt_tokens"] > 0
        assert speculative["termination_reason"] in ("eos", "length")
        assert speculative["path_stats"]["speculative_used"] == 1
        assert speculative["kv_positions"] == (
            speculative["prompt_tokens"] + len(speculative["tokens"]) - 1)
    finally:
        target.close()
        draft.close()


def test_dense_speculation_falls_back_exactly_when_prompt_exceeds_draft_vocab(tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.speculative import SpeculativeDecoder

    _build_dense_fixture(tmp_path)
    target = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, governor=False))
    draft = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, governor=False))
    try:
        prompt = "draft vocabulary boundary"
        baseline = target.generate(prompt, max_tokens=8)
        decoder = SpeculativeDecoder(target, draft, k=3)
        # The physical fixture has all rows, but the declared bound exercises
        # the production case where the target tokenizer has added IDs the
        # smaller Qwen draft does not own.
        draft.cfg.vocab_size = 1
        speculative = decoder.generate(prompt, max_tokens=8)

        assert speculative["tokens"] == baseline["tokens"]
        assert speculative["stats"].draft_oov_fallbacks == 1
        assert speculative["stats"].proposed == 0
    finally:
        target.close()
        draft.close()


def test_resident_draft_chain_keeps_target_tokens_exact(tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.speculative import SpeculativeDecoder

    _build_dense_fixture(tmp_path)
    target = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, governor=False))
    draft = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, resident_fast_decode=True, governor=False))
    try:
        prompt = "one-sync resident draft chain"
        baseline = target.generate(prompt, max_tokens=12)
        speculative = SpeculativeDecoder(target, draft, k=4).generate(
            prompt, max_tokens=12)

        assert speculative["tokens"] == baseline["tokens"]
        assert speculative["stats"].resident_draft_rounds > 0
        assert speculative["stats"].resident_draft_tokens > 0
    finally:
        target.close()
        draft.close()


def test_speculative_stream_callbacks_reconstruct_exact_text(tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.speculative import SpeculativeDecoder

    _build_dense_fixture(tmp_path)
    target = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, governor=False))
    draft = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, resident_fast_decode=True, governor=False))
    try:
        prompt = "stream target-verified tokens"
        baseline = target.generate(prompt, max_tokens=12)
        chunks = []
        progress = []
        speculative = SpeculativeDecoder(target, draft, k=4).generate(
            prompt, max_tokens=12, on_token=chunks.append,
            on_progress=progress.append)

        assert speculative["tokens"] == baseline["tokens"]
        assert "".join(chunks) == speculative["text"]
        assert progress == [{
            "phase": "prefill",
            "completed_tokens": speculative["prompt_tokens"],
            "total_tokens": speculative["prompt_tokens"],
            "cache_source": "speculative-cold",
        }]
    finally:
        target.close()
        draft.close()


def test_speculative_string_stop_matches_target_text_tokens_and_kv(tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.speculative import SpeculativeDecoder

    _build_dense_fixture(tmp_path)
    target = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, governor=False))
    draft = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, resident_fast_decode=True, governor=False))
    try:
        prompt = "speculative stop boundary"
        truth = target.generate(prompt, max_tokens=14)
        decoded = target.tokenizer.decode(truth["tokens"])
        stop = decoded[3:6]
        assert stop
        baseline = target.generate(prompt, max_tokens=14, stop=[stop])
        chunks = []
        speculative = SpeculativeDecoder(target, draft, k=6).generate(
            prompt, max_tokens=14, stop=[stop], on_token=chunks.append)

        assert speculative["tokens"] == baseline["tokens"]
        assert speculative["text"] == baseline["text"] == "".join(chunks)
        assert speculative["termination_reason"] == "stop_sequence"
        assert speculative["stop_sequence"] == stop
        assert speculative["kv_positions"] == (
            speculative["prompt_tokens"] + len(speculative["tokens"]) - 1)
    finally:
        target.close()
        draft.close()


def test_speculative_accepted_eos_trims_rest_of_verify_window(tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.speculative import SpeculativeDecoder

    _build_dense_fixture(tmp_path)
    target = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, governor=False))
    draft = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, resident_fast_decode=True, governor=False))
    try:
        prompt = "speculative eos rollback"
        truth = target.generate(prompt, max_tokens=14)
        eos = truth["tokens"][3]
        target.cfg.eos_token_ids = (eos,)
        baseline = target.generate(prompt, max_tokens=14)
        speculative = SpeculativeDecoder(target, draft, k=6).generate(
            prompt, max_tokens=14)

        assert speculative["tokens"] == baseline["tokens"]
        assert speculative["termination_reason"] == "eos"
        assert speculative["kv_positions"] == (
            speculative["prompt_tokens"] + len(speculative["tokens"]) - 1)
    finally:
        target.close()
        draft.close()


def test_speculative_exact_prompt_reuses_paired_target_and_draft_kv(tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.speculative import SpeculativeDecoder

    _build_dense_fixture(tmp_path)
    target = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, governor=False))
    draft = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100, resident_fast_decode=True, governor=False))
    try:
        prompt = "paired speculative prompt cache"
        baseline = target.generate(prompt, max_tokens=12)
        decoder = SpeculativeDecoder(
            target, draft, k=4, prompt_cache_min_tokens=1)
        first = decoder.generate(prompt, max_tokens=12)
        second = decoder.generate(prompt, max_tokens=12)
        continued_prompt = prompt + " continued"
        continued_baseline = target.generate(continued_prompt, max_tokens=12)
        continued = decoder.generate(continued_prompt, max_tokens=12)

        assert first["tokens"] == second["tokens"] == baseline["tokens"]
        assert not first["path_stats"]["prompt_cache_exact_hit"]
        assert second["path_stats"]["prompt_cache_exact_hit"] == 1
        assert second["path_stats"]["prompt_cache_prefix_tokens"] == (
            second["prompt_tokens"])
        assert second["path_stats"]["prompt_cache_source"] == (
            "speculative-memory")
        assert second["kv_positions"] == (
            second["prompt_tokens"] + len(second["tokens"]) - 1)
        assert continued["tokens"] == continued_baseline["tokens"]
        assert not continued["path_stats"]["prompt_cache_exact_hit"]
        assert continued["path_stats"]["prompt_cache_prefix_tokens"] == 0
        assert continued["path_stats"]["prompt_cache_source"] == (
            "speculative-cold")
    finally:
        target.close()
        draft.close()


def test_pipelined_lookahead_rolls_back_when_eos_stops_early(tmp_path):
    _build_dense_fixture(tmp_path)
    truth, _ = _generate(tmp_path, True, 10)
    eos = truth["tokens"][3]
    expected = truth["tokens"][:truth["tokens"].index(eos) + 1]

    slow, slow_offset = _generate(tmp_path, False, 10, eos=(eos,))
    fast, fast_offset = _generate(tmp_path, True, 10, eos=(eos,))

    assert fast["tokens"] == slow["tokens"] == expected
    assert fast["termination_reason"] == "eos"
    expected_offset = fast["prompt_tokens"] + len(expected) - 1
    assert fast_offset == slow_offset == expected_offset


def test_pipelined_lookahead_rolls_back_on_string_stop(tmp_path):
    _build_dense_fixture(tmp_path)
    truth, _ = _generate(tmp_path, True, 10)
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    stop = tokenizer.decode(truth["tokens"][:4])
    assert stop

    slow, slow_offset = _generate(tmp_path, False, 10, stop=[stop])
    fast, fast_offset = _generate(tmp_path, True, 10, stop=[stop])

    assert fast["tokens"] == slow["tokens"] == truth["tokens"][:4]
    assert fast["termination_reason"] == "stop_sequence"
    expected_offset = fast["prompt_tokens"] + len(fast["tokens"]) - 1
    assert fast_offset == slow_offset == expected_offset


def test_lossy_fused_swiglu_profile_is_deterministic_and_reported(tmp_path):
    _build_dense_fixture(tmp_path)
    first, _ = _generate(tmp_path, True, 12, fused_swiglu=True)
    second, _ = _generate(tmp_path, True, 12, fused_swiglu=True)

    assert first["tokens"] == second["tokens"]
    assert first["path_stats"]["fused_swiglu"] == 1


def test_portable_schedule_evaluates_prompt_endpoint_separately(tmp_path):
    _build_dense_fixture(tmp_path)
    result, offset = _generate(
        tmp_path, True, 12, last_token_separate=True)

    assert result["path_stats"]["prefill_chunks"] == 1
    assert offset == result["prompt_tokens"] + len(result["tokens"]) - 1


def test_unquantized_tied_head_fallback_is_not_truth_tested(tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine

    _build_dense_fixture(tmp_path)
    engine = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100,
        quant_bits=4,
        quant_group_size=32,
        quant_mode="mxfp4",
        quant_min_dim=10_000,  # policy deliberately returns the raw MLX array
        quantize_tied_lm_head=True,
        governor=False,
    ))
    try:
        assert engine.generate("array fallback", max_tokens=2)["tokens"]
    finally:
        engine.close()


def test_tied_head_quantization_fetches_unpinned_embedding(tmp_path):
    from runtime import quant
    from runtime.engine import RuntimeConfig, StreamingEngine

    _build_dense_fixture(tmp_path)
    engine = StreamingEngine(str(tmp_path), RuntimeConfig(
        max_weight_cache_mb=100,
        pin_embeddings=False,
        quant_bits=4,
        quant_group_size=32,
        quant_mode="mxfp4",
        quant_min_dim=0,
        quantize_tied_lm_head=True,
        governor=False,
    ))
    try:
        assert isinstance(engine._tied_lm_head_w, quant.QTensor)
        assert engine.generate("unpinned embedding", max_tokens=2)["tokens"]
    finally:
        engine.close()
