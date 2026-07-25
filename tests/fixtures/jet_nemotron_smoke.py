"""Jet-Nemotron end-to-end smoke test / speed benchmark, real checkpoint.

Not part of the StreamingEngine/server serving path yet (see
docs/future_lossless_techniques.md's Jet-Nemotron entry for why: the
"jet"/"swa"/"attn" block math is oracle-verified and this script proves it
runs correctly end-to-end against real released weights, but full
engine.py/server.py wiring -- paging, hot-KV, request handling -- is a
separate, larger step not attempted here). Loads the whole checkpoint
directly via mx.load (feasible at ~7.4GB/3.7GB for the 4B/2B sizes on a
16GB machine for a short, one-off benchmark run; NOT how a persistent
server should load a model).

Usage:
  .venv/bin/python tests/fixtures/jet_nemotron_smoke.py \
      --model models/Jet-Nemotron-4B --prompt "..." --max-tokens 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from runtime.config import ModelConfig
from runtime.jet_nemotron import run_jet_nemotron_block
from runtime.kda_state import KDAStateCache
from runtime.kv_cache import KVCache
from runtime.layer_runner import embed, final_logits


def load_weights(model_dir: Path) -> dict[str, mx.array]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        shard_names = ["model.safetensors"]
    weights: dict[str, mx.array] = {}
    for shard in shard_names:
        weights.update(mx.load(str(model_dir / shard)))
    return weights


def build_kv(cfg: ModelConfig) -> KVCache:
    kv = KVCache(cfg.num_hidden_layers)
    if "jet" in cfg.layer_types:
        kv.kda_cache = KDAStateCache(cfg.num_hidden_layers)
    return kv


def forward(tokens: list[int], kv: KVCache, weights: dict, cfg: ModelConfig,
            offset: int) -> mx.array:
    x = embed(mx.array(tokens), weights["model.embed_tokens.weight"])
    for layer in range(cfg.num_hidden_layers):
        x = run_jet_nemotron_block(
            x, weights, f"model.layers.{layer}", cfg, kv, layer, offset)
    return final_logits(x, weights["model.norm.weight"],
                         weights["model.embed_tokens.weight"], cfg.rms_norm_eps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    cfg = ModelConfig.from_dir(args.model)
    print(f"loaded config: {cfg.num_hidden_layers} layers, "
          f"{cfg.layer_types.count('jet')} jet / "
          f"{cfg.layer_types.count('swa')} swa / "
          f"{cfg.layer_types.count('attn')} attn")

    t0 = time.perf_counter()
    weights = load_weights(args.model)
    print(f"loaded {len(weights)} tensors in {time.perf_counter() - t0:.2f}s")

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    ids = tokenizer.encode(args.prompt).ids
    print(f"prompt: {len(ids)} tokens")

    kv = build_kv(cfg)
    t0 = time.perf_counter()
    logits = forward(ids, kv, weights, cfg, offset=0)  # (vocab,) -- last position only
    next_tok = int(mx.argmax(logits).item())
    prefill_s = time.perf_counter() - t0
    print(f"prefill: {prefill_s:.3f}s ({len(ids) / prefill_s:.1f} tok/s)")

    generated = [next_tok]
    t0 = time.perf_counter()
    pos = len(ids)
    for _ in range(args.max_tokens - 1):
        logits = forward([generated[-1]], kv, weights, cfg, offset=pos)
        next_tok = int(mx.argmax(logits).item())
        generated.append(next_tok)
        pos += 1
    decode_s = time.perf_counter() - t0
    n_decoded = len(generated) - 1
    if n_decoded > 0:
        print(f"decode: {decode_s:.3f}s for {n_decoded} tokens "
              f"({n_decoded / decode_s:.1f} tok/s)")

    text = tokenizer.decode(generated)
    print(f"output: {text!r}")


if __name__ == "__main__":
    main()
