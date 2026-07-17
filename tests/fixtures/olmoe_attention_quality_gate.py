#!/usr/bin/env python3
"""Real-checkpoint quality/speed gate for hybrid OLMoE attention.

Uses only repository text and embedded multiple-choice questions. The baseline
and candidate share expert weights, router, embedding, and exact BF16 head; only
the resident attention matrices differ.

    python tests/fixtures/olmoe_attention_quality_gate.py MODEL \
      --profile mxfp8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.quant import QuantPolicy


CHOICES = (
    ("What is 7 times 8?", ("54", "56", "64", "48"), 1),
    ("What is the square root of 81?", ("7", "8", "9", "10"), 2),
    ("Which number is prime?", ("15", "21", "17", "27"), 2),
    ("What is 15 minus 6?", ("7", "8", "10", "9"), 3),
    ("What comes next: 2, 3, 5, 8, 13?", ("18", "19", "20", "21"), 3),
    ("What is the chemical formula for water?", ("CO2", "H2O", "O2", "NaCl"), 1),
    ("Which planet is called the Red Planet?", ("Venus", "Mars", "Jupiter", "Mercury"), 1),
    ("Which gas do plants absorb?", ("Oxygen", "Nitrogen", "Carbon dioxide", "Helium"), 2),
    ("Water boils at sea level at what Celsius temperature?", ("50", "75", "100", "212"), 2),
    ("Which force attracts objects toward Earth?", ("Magnetism", "Gravity", "Friction", "Electricity"), 1),
    ("Binary search has what time complexity?", ("O(1)", "O(log n)", "O(n)", "O(n^2)"), 1),
    ("Which data structure is FIFO?", ("Stack", "Tree", "Queue", "Heap"), 2),
    ("HTTP status 404 means what?", ("Success", "Unauthorized", "Not Found", "Server Error"), 2),
    ("What is len(set([1,2,2,3]))?", ("2", "3", "4", "1"), 1),
    ("A mutex primarily provides what?", ("Compression", "Mutual exclusion", "Sorting", "Serialization"), 1),
    ("Which word is a synonym for rapid?", ("slow", "fast", "quiet", "heavy"), 1),
    ("Which word is the opposite of scarce?", ("rare", "small", "abundant", "empty"), 2),
    ("What is the Spanish word for cat?", ("perro", "gato", "casa", "agua"), 1),
    ("What is the plural of mouse?", ("mouses", "mouse", "mice", "meese"), 2),
    ("Which is a noun?", ("quickly", "blue", "happiness", "run"), 2),
    ("What is the capital of France?", ("Berlin", "Paris", "Rome", "Madrid"), 1),
    ("What is the capital of Japan?", ("Seoul", "Beijing", "Tokyo", "Bangkok"), 2),
    ("The Nile is primarily on which continent?", ("Asia", "Africa", "Europe", "South America"), 1),
    ("Which ocean is largest?", ("Atlantic", "Indian", "Arctic", "Pacific"), 3),
    ("Toronto is in which country?", ("USA", "Canada", "Australia", "UK"), 1),
    ("All whales are mammals; all mammals are warm-blooded. Whales are?", ("cold-blooded", "warm-blooded", "fish", "plants"), 1),
    ("Rain implies wet ground; the ground is not wet. What follows?", ("it rained", "it did not rain", "it snowed", "nothing"), 1),
    ("Which does not belong?", ("apple", "banana", "carrot", "orange"), 2),
    ("A triangle has how many sides?", ("2", "3", "4", "5"), 1),
    ("If x=3, what is 2x+1?", ("5", "6", "7", "8"), 2),
)


def _candidate_layers(engine: StreamingEngine, profile: str):
    q8 = QuantPolicy(
        bits=8, group_size=32, mode="mxfp8", min_dim=0,
        quantize_attention=True, quantize_mlp=False,
        quantize_router=False, quantize_lm_head=False)
    q4 = QuantPolicy(
        bits=4, group_size=32, mode="mxfp4", min_dim=0,
        quantize_attention=True, quantize_mlp=False,
        quantize_router=False, quantize_lm_head=False)
    nv4 = QuantPolicy(
        bits=4, group_size=16, mode="nvfp4", min_dim=0,
        quantize_attention=True, quantize_mlp=False,
        quantize_router=False, quantize_lm_head=False)
    affine6 = QuantPolicy(
        bits=6, group_size=64, mode="affine", min_dim=0,
        quantize_attention=True, quantize_mlp=False,
        quantize_router=False, quantize_lm_head=False)
    q4_suffixes = {
        "mxfp8-v-o-mxfp4": ("v_proj.weight", "o_proj.weight"),
        "mxfp4": ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"),
    }.get(profile, ())
    nv4_suffixes = {
        "mxfp8-v-o-nvfp4": ("v_proj.weight", "o_proj.weight"),
        "mxfp8-v-nvfp4": ("v_proj.weight",),
        "mxfp8-o-nvfp4": ("o_proj.weight",),
        "nvfp4": ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"),
    }.get(profile, ())
    affine6_suffixes = {
        "affine6-g64": (
            "q_proj.weight", "k_proj.weight",
            "v_proj.weight", "o_proj.weight"),
        "mxfp8-v-o-affine6": ("v_proj.weight", "o_proj.weight"),
        "mxfp8-q-k-affine6": ("q_proj.weight", "k_proj.weight"),
        "mxfp8-q-affine6": ("q_proj.weight",),
        "mxfp8-k-affine6": ("k_proj.weight",),
    }.get(profile, ())
    layers = []
    for trunk, experts in engine._resident_moe_layers:
        transformed = {}
        for name, weight in trunk.items():
            if (".self_attn." in name and name.endswith(".weight")
                    and getattr(weight, "ndim", 0) == 2):
                policy = (
                    q4 if name.endswith(q4_suffixes) else
                    nv4 if name.endswith(nv4_suffixes) else
                    affine6 if name.endswith(affine6_suffixes) else
                    q8
                )
                weight = policy.transform(name, weight)
            transformed[name] = weight
        layers.append((transformed, experts))
    return tuple(layers)


def _score_choices(engine: StreamingEngine, layers) -> tuple[int, list[int]]:
    engine._resident_moe_layers = layers
    label_ids = [engine.tokenizer.encode(" " + value).ids[0] for value in "ABCD"]
    predictions = []
    correct = 0
    for question, choices, answer in CHOICES:
        prompt = question + "\n" + "\n".join(
            f"{label}. {choice}" for label, choice in zip("ABCD", choices)
        ) + "\nAnswer:"
        tokens = engine.tokenizer.encode(prompt).ids
        kv = engine.new_kv()
        logits = engine.forward_tokens(tokens, kv)[-1]
        prediction = int(mx.argmax(logits[mx.array(label_ids)]))
        predictions.append(prediction)
        correct += prediction == answer
    return correct, predictions


def _corpus_chunks(engine: StreamingEngine) -> list[list[int]]:
    paths = (
        ROOT / "README.md", ROOT / "runtime/engine.py",
        ROOT / "runtime/server.py", ROOT / "runtime/layer_runner.py",
        ROOT / "docs/memory_model.md", ROOT / "tests/test_server_pure.py",
    )
    chunks = []
    for path in paths:
        tokens = engine.tokenizer.encode(path.read_text()).ids
        for fraction in (0.1, 0.55):
            start = min(
                max(0, int(len(tokens) * fraction)),
                max(0, len(tokens) - 257),
            )
            chunk = tokens[start:start + 257]
            if len(chunk) == 257:
                chunks.append(chunk)
    return chunks


def _score_nll(engine: StreamingEngine, layers, chunks) -> tuple[float, list[int]]:
    engine._resident_moe_layers = layers
    total_nll = 0.0
    total_tokens = 0
    top_tokens = []
    for chunk in chunks:
        kv = engine.new_kv()
        logits = engine.forward_tokens(chunk[:-1], kv).astype(mx.float32)
        targets = mx.array(chunk[1:])
        selected = mx.take_along_axis(
            logits, targets[:, None], axis=-1).squeeze(-1)
        nll = mx.logsumexp(logits, axis=-1) - selected
        top = mx.argmax(logits, axis=-1)
        mx.eval(nll, top)
        total_nll += float(mx.sum(nll))
        total_tokens += len(targets)
        top_tokens.extend(int(value) for value in top.tolist())
    return total_nll / total_tokens, top_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--profile", choices=(
            "mxfp8", "mxfp8-v-o-mxfp4", "mxfp4",
            "mxfp8-v-o-nvfp4", "mxfp8-v-nvfp4",
            "mxfp8-o-nvfp4", "nvfp4",
            "affine6-g64",
            "mxfp8-v-o-affine6", "mxfp8-q-k-affine6",
            "mxfp8-q-affine6", "mxfp8-k-affine6",
        ),
        default="mxfp8")
    parser.add_argument("--max-nll-regression", type=float, default=0.04)
    parser.add_argument("--max-choice-regression", type=int, default=0)
    args = parser.parse_args()

    rc = RuntimeConfig(
        max_weight_cache_mb=7000, pin_lm_head=True,
        quant_bits=4, quant_group_size=32, quant_mode="mxfp4", quant_min_dim=0,
        quant_attention=False, quant_router=False, quant_lm_head=False,
        resident_moe_decode=True, fused_swiglu=True,
        rerank_lm_head=True, rerank_lm_head_candidates=32,
        prefill_chunk_size=2048, prefill_last_token_separate=True,
        governor=True)
    engine = StreamingEngine(args.model, rc)
    baseline_layers = engine._resident_moe_layers
    candidate_layers = _candidate_layers(engine, args.profile)
    reranked_head = engine._lm_head_w
    engine._lm_head_w = reranked_head.exact
    try:
        chunks = _corpus_chunks(engine)
        base_correct, base_predictions = _score_choices(engine, baseline_layers)
        candidate_correct, candidate_predictions = _score_choices(
            engine, candidate_layers)
        base_nll, base_top = _score_nll(engine, baseline_layers, chunks)
        candidate_nll, candidate_top = _score_nll(
            engine, candidate_layers, chunks)
    finally:
        engine._lm_head_w = reranked_head
        engine._resident_moe_layers = baseline_layers
        engine.close()
        mx.clear_cache()

    total = len(base_top)
    result = {
        "profile": args.profile,
        "choice_baseline": base_correct,
        "choice_candidate": candidate_correct,
        "choice_agreement": sum(
            left == right
            for left, right in zip(base_predictions, candidate_predictions)),
        "choice_total": len(CHOICES),
        "baseline_nll": base_nll,
        "candidate_nll": candidate_nll,
        "nll_regression": candidate_nll - base_nll,
        "baseline_ppl": math.exp(base_nll),
        "candidate_ppl": math.exp(candidate_nll),
        "top1_agreement": sum(
            left == right for left, right in zip(base_top, candidate_top)) / total,
        "corpus_tokens": total,
    }
    print(json.dumps(result, indent=2), flush=True)
    if candidate_correct < base_correct - args.max_choice_regression:
        raise SystemExit(1)
    if candidate_nll - base_nll > args.max_nll_regression:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
