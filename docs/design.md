# Virtualized LLM Runtime — Phase 0 Design Note

Date: 2026-07-09. Target machine: M4 Mac Mini, 16 GB unified memory, macOS 26.5.
Models live on an external USB drive, ~189 MB/s measured write throughput;
internal SSD has <10 GB free and must not be used.

## Chosen model format: safetensors, loaded via `mx.load()`

Findings from format investigation:

| Option | Verdict |
|---|---|
| `mx.load(path)` (MLX native safetensors reader) | **Chosen.** Fully lazy per tensor: `mx.load` on a 257 MB file returns in 3 ms with 0 MB materialized; `mx.eval(one_tensor)` reads only that tensor (2 MB active); bf16 supported natively. Dropping references + `mx.clear_cache()` returns Metal memory to 0. |
| `safetensors.safe_open(framework="mlx")` | **Broken for bf16** in safetensors 0.8.0: raises `TypeError: data type 'bfloat16' not understood` (routes through numpy). Nearly all modern checkpoints are bf16, so this is disqualifying. |
| `safetensors.safe_open(framework="numpy")` | Same bf16 problem. |
| GGUF | Single-file, quantized-block layout; MLX can read it but per-tensor lazy access + dequant control is messier. Deferred to a later phase for quantized runs (see `formats/gguf_notes.md` when written). |
| MLX-native converted weights (`mlx_lm.convert`) | Also safetensors under the hood; conversion step duplicates disk usage for no Phase-1 benefit. Not needed. |

Sharded checkpoints (`model-0000X-of-0000Y.safetensors` + `model.safetensors.index.json`)
are handled by reading the index JSON to map tensor name → shard file, then `mx.load`
per shard. Single-file models synthesize the same mapping.

## Loading one transformer block at a time

The mechanism, verified empirically:

1. `mx.load(shard)` → dict of *lazy* file-backed `mx.array`s (costs ~0).
2. Collect the ~9 tensors for `model.layers.i.*`, `mx.eval()` them → only those bytes
   are read from disk and materialized in unified memory.
3. Run the block forward.
4. Drop every reference to the layer's arrays and call `mx.clear_cache()` → memory
   returns to the MLX allocator and the OS.

Caveat: once evaluated, an array in the `mx.load` dict *holds* its data. Eviction
therefore means dropping the dict entries; re-fetch re-calls `mx.load` (3 ms) for
fresh lazy handles. The loader treats `mx.load` as a cheap open, not a load.

## Memory layout / tiers on Apple Silicon

Unified memory collapses the classic "CPU RAM vs GPU VRAM" distinction — there is no
host↔device copy. The real hierarchy here is:

```
Tier 0/1 (merged): materialized mx.arrays in unified memory (wired, Metal-visible)
Tier 2a: macOS file page cache (clean pages of the safetensors files — free reloads)
Tier 2b: external SSD (cold reads, ~200 MB/s on this USB drive)
Tier 3: (later) network storage
```

Consequences:
- "Move to compute device" is a no-op; placement policy is purely *materialized vs not*.
- The OS page cache silently accelerates re-reads. Good for speed, bad for honest
  benchmarks — cold-cache measurements need `F_NOCACHE`/`purge` discipline (Phase 10).
- Disk throughput (~200 MB/s here) sets the floor for streamed token latency:
  a 14 GB fp16 7B model ⇒ ~70 s/token if every layer is read cold every token.
  Weight cache + prefetch + page cache are what claw this back.

## What MLX cannot do cleanly (documented gaps)

- No public API to pin/wire specific arrays or control residency beyond ref-holding.
- No async/streamed disk→GPU read API; prefetch must be built with threads doing
  `mx.eval` (or plain file reads to warm the page cache) off the main thread.
- Lazy evaluation makes naive timing wrong; every measurement must bracket with
  explicit `mx.eval`/`mx.synchronize`.
- `mx.get_active_memory()` tracks Metal allocations only; process RSS (psutil) is
  tracked alongside it in telemetry.

## First target models

1. **SmolLM2-135M** (Llama arch, 30 layers, hidden 576, GQA 9/3 heads, tied embeddings,
   bf16, single shard) — correctness bring-up.
2. TinyLlama-1.1B or Qwen2.5-1.5B — second architecture datapoint (Qwen2 adds attention bias).
3. A 7B-class model in fp16 (~14 GB > usable RAM) — first "can't fit normally" demo.

Architecture scope for Phase 1: Llama-family decoder (RMSNorm, RoPE, GQA, SwiGLU),
which covers all three targets. Config is read from `config.json` (HF schema).

## Phase 1 execution plan

- Prefill runs layer-at-a-time over the *whole prompt* (one disk sweep per prompt),
  then decode sweeps all layers once per token.
- Resident set during decode: embeddings (reused as lm_head when tied), one layer's
  weights, final norm, KV cache, activations.
- Greedy decoding only. Telemetry logs per-layer load/compute time, RSS, Metal
  active/peak memory every token.
