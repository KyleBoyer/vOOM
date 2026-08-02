# Kimi K3 prefill findings

Measured 2026-08-01 on the project M4 Mac mini (16 GB unified memory).  This
document separates exact model computation, algebraically equivalent but
floating-point-reassociated computation, and explicitly lossy side-quest
profiles.

## AirLLM review

AirLLM's useful Kimi K3 changes are checkpoint-facing rather than a substitute
prefill engine:

- normalize the released multimodal wrapper's `language_model.*`,
  `mm_projector.*`, and `vision_tower.*` names;
- read compressed-tensors metadata nested under `text_config`;
- retain packed expert weights and uint8 scales in their released dtypes;
- fetch individual experts directly; and
- expose top-level AttnRes and vision modules.

vOOM retains those capabilities while reading the released safetensors index
and source shards directly.  It does not create AirLLM's second split
checkpoint tree.  `tests/test_airllm_k3_capabilities.py` pins this contract.

## Implemented prefill work

The following operations are exact tiering/scheduling changes:

- layer-stationary weight reuse;
- compressed and weight-absorbed MLA with tiled online softmax;
- streamed/tiled AttnRes;
- in-place reuse of the prompt activation/residual buffer;
- tiled shared-expert computation and accumulation into the routed output;
- exact BF16 AttnRes snapshots spilled in groups of four;
- exact FP32/BF16 KDA endpoints and released-dtype MLA latents spilled after
  prefill and lazily restored for decode;
- Darwin `F_NOCACHE` on the ephemeral spill descriptors; and
- compiled 32-position MLX KDA segments that preserve the ordinary MLX
  operators, reduction order, and state boundaries.

The native serial KDA Metal kernel implements the released recurrence algebra,
but reassociates FP32 reductions.  It is therefore classified as lossy for a
byte-exact released-model claim.  REAP50 expert pruning and routed top-4 in
place of released top-16 are independently lossy.

The engine also retires a secondary 400 MB recurring-layer uncertainty pad
after one matching layer signature completes.  The independent 1.2 GB memory
governor reserve remains active.  Prompt-length scheduling uses the production
memory-retry contract; a failed prefill can replay at 128, 32, 8, then one
position only before sampling begins.

## Complete real harness result

The complete 93-layer gate preserved the 178,616-byte capture, all three
messages, all 134 tools in request order, temperature 1, streaming, and
automatic tool choice.  The capture SHA-256 is
`8ac18b8e8bc190180b4cc0e02c2453d313ec850642cc5d5f63b32e5537b90e85`;
the 46,107-token rendered prompt SHA-256 is
`0bdac81af27de1b36b16a6c937e67be60c2ce0b565bc60c02ce91ebf0f4f04f1`.
Only the local model and explicit two-token output cap changed.

| profile / cache state | first token | next token | total | peak Metal | class |
|---|---:|---:|---:|---:|---|
| REAP50 + top-4 + native KDA, fresh engine, prompt miss | **149.765 min** | **91.831 s** | **151.296 min** model time | **7.184 GB** | measured, lossy |
| same gate including process/envelope overhead | about **151.0 min** | 91.831 s | **152.516 min** wall | 7.184 GB | first-token split derived |
| released lossless, unseen prompt | about **276 min / 4.6 h** | about 2.7 min/token | not measured | not gated | projected |
| exact full-prompt cache hit | not measured | schedule-dependent | not measured | not measured | exact only in matching namespace |

The measured run used a fresh engine and had no prompt/KV cache hit.  It is not
a strict cold-model-storage number: mapped model pages may have had mixed
macOS page-cache residency, and logical weight-view counters cannot prove
physical reads.  The ephemeral state tier was independently uncached:
`F_NOCACHE` covered 22 AttnRes, 552 KDA, and 48 MLA descriptors.  It wrote
13.220 GB of AttnRes snapshots and read 551.266 GB through the tiled consumer;
KDA wrote/read 449.4 MB, and MLA wrote/read 1.281 GB.

The released-lossless estimate applies the measured 4K five-layer class ratios
(1.9544x KDA and 1.7202x full attention) to the complete fast gate's measured
69-KDA-layer and 24-full-attention-layer totals.  It is not a substitute for a
completed released-model gate.

The completed fast run therefore does not yet reach tens of minutes for a
never-seen 46K-token request.  An exact cache may reuse only a byte-identical
prompt or prefix under the same model, tokenizer, arithmetic, and schedule
namespace.  Every uncached suffix position must still cross every released
layer.  MTP/speculative decoding can reduce later-token latency, not the first
cold prefill.

## Native KDA microprobe

At K3's `[B=1,H=96,L=256,K=128,V=128]` FP32 shape, the native serial kernel ran
in 0.022318 s versus 0.194387 s for the ordinary serial MLX path: **8.7098x**.
Maximum output/state error was `3.73e-8` / `8.94e-8`.  The compiled 32-position
MLX path is the byte-identical alternative.

The primary local artifacts are
`logs/k3_kai_retry128_top4_reap50_93l_20260801.json`,
`logs/gates/k3_kai_retry128_top4_reap50_93l_20260801/`, and
`logs/k3_native_scan_probe_20260801.json`.  The tile-256 negative is retained
under `logs/gates/k3_kai_final_top4_reap50_93l_20260801/`; the governor refused
layer 32 rather than cross the live memory reserve.
