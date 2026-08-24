# Huihui Qwen3.8 27B: FreeToken audit and measured improvements

Date: 2026-08-24

This note records the FreeToken-inspired work on the explicit lossy Huihui
Qwen3.8 27B agent profile. The served MXFP4 target and released-BF16 MTP
sidecar are unchanged. Any optimization that did not pass exact-output and
timing gates was reverted.

## Accepted: exact BF16 MTP split across both SSDs

`runtime/qwen_mtp_bf16_fast_tier.py` copies a selected set of **complete BF16
tensors byte-for-byte** from the released MTP sidecar to the internal fast
tier. The other tensors remain on the external NVMe. `WeightStore` can fetch
the two sets concurrently because the devices have different `st_dev` values.
The manifest is committed last and is bound to the source metadata, the full
fast-file SHA-256, tensor names, shapes, dtypes, offsets, and sizes. Invalid or
stale metadata fails closed.

The exhaustive 15-tensor partition selected 471,911,424 bytes internally and
377,487,360 bytes externally (about 56/44, based on the measured 2.02/1.60
GB/s device rates). The resulting global internal fast tier is 32,470,958,356
bytes, with 61,855,973,376 bytes free: below the 90 GB cap and above the 10 GB
free-space floor.

Cold A/B used the unmodified 178,616-byte, 134-tool captured request, streamed,
with only the documented 16-token benchmark cap. Both arms used the same
temporary 5.5 GB available/post-generation benchmark floors because the
machine could not admit the run with the shipped 6 GB floor at the time. The
profile's shipped 6 GB safety floor was not changed.

| Metric | External-only MTP | Exact two-SSD MTP | Change |
|---|---:|---:|---:|
| MTP sidecar load | 4.149 s | 2.615 s | -36.96% |
| MTP draft work | 5.244 s | 3.554 s | -32.21% |
| Decode | 63.528 s | 61.295 s | -3.51% |
| Engine | 122.945 s | 119.983 s | -2.41% |
| Wall | 126.123 s | 123.424 s | -2.14% |
| Peak Metal | 3.180 GB | 3.199 GB | +19 MB |

Both outputs had SHA-256
`9b170095a170f34a248af480bab274aef28f42ebf1504196bc28cc57e5b87b81`.
The accepted arm read 65.571 GB from the internal tier and 57.698 GB from the
external tier over 584 parallel fetches; 34.702 seconds of device service were
overlapped.

Artifacts:

- `logs/freetoken_mtp_bf16_fast_tier_build.json`
- `logs/freetoken_mtp_external_confirm_floor5500_capture16.json`
- `logs/freetoken_mtp_split_floor5500_capture16.json`

## Verifier productivity is now explicit

The Qwen MTP target verifier does more than validate proposals. When a draft
prefix is accepted, its last target position supplies the authoritative bonus
next token; exact KV and DeltaNet/KDA state are committed only through the
accepted prefix, with rejected positions rolled back. The verifier is not a
second approximate predictor and does not alter the target distribution.

The capture used 8 target sweeps for 15 post-bootstrap tokens: 8 accepted MTP
draft tokens plus 7 verifier bonus tokens. This avoided 7 full target sweeps
and achieved 1.875 output tokens per target sweep. New telemetry separates
input, committed and rolled-back verifier positions, accepted/correction/bonus
tokens, target sweeps avoided, tokens per sweep, and draft/verifier time.

## Dual-disk and prefetch instrumentation

New request telemetry reports per-device bytes, service time, parallel-fetch
wall time and the hidden overlap lower bound. It also fixes external raw
safetensors accounting, which previously omitted those bytes. Prefetch now
reports loads, useful and wasted pages/bytes/load time, waits, and the useful
load time hidden before demand.

On the accepted capture, all 19 prefetched pages were consumed, none waited or
were wasted, and at least 1.966 seconds of load time was hidden. The current
prefetch depth of two remains the measured choice.

## Rejected: moving more target-body tensors to the external SSD

`runtime/qwen_fast_tier_rebalance.py` tested reducing the internal trunk share
from 53.06% (6.866 GB) to 39.97% (5.173 GB) without changing or deleting tensor
bytes. The response hash and 8/8 MTP acceptance were preserved, but internal
service time barely moved (60.209 to 59.874 seconds), external service rose,
and decode regressed to 65.902 seconds in a lower-memory run. The candidate
body manifest was rejected and the original active manifest was restored.

This shows that body-tier time is currently dominated by per-fetch loading and
materialization as well as raw bytes; blindly moving more body bytes is not an
improvement.

## Rejected: CPU dense projection offload

`runtime/free_token_cpu_probe.py` tested the released Qwen gate shape
`(1,5120) @ (17408,5120)` in BF16 with 0%, 12.5%, 25%, 50%, and 100% of output
rows on the CPU. Every split was byte-identical to full Metal, but every
non-zero CPU split was slower:

| CPU rows | Median | Speed vs. full Metal |
|---:|---:|---:|
| 0% | 2.230 ms | 1.000x |
| 12.5% | 3.100 ms | 0.719x |
| 25% | 4.105 ms | 0.543x |
| 50% | 5.981 ms | 0.373x |
| 100% | 8.594 ms | 0.259x |

CPU remains useful for request orchestration, hashing, decompression and disk
workers, but not dense per-token GEMMs on this unified-memory M4. FreeToken's
bandwidth-adaptive `q*` CPU/GPU policy targets cache-missed MoE experts across
PCIe; Huihui Qwen3.8 is dense and Apple CPU/Metal share the same DRAM fabric.
Artifact: `logs/freetoken_cpu_offload_probe.json`.

## FreeToken technique mapping

FreeToken's double-buffered layer streaming, global cache, elastic memory
policy and recurrent/KV semantic anchors map to facilities already present in
vOOM: exact two-device async fetch, bounded LRU/pinning and prefetch, the
memory governor, and exact persistent hybrid KDA/KV prefix checkpoints. Its
final-tensor storage principle matches the raw fast-tier layer containers.
The useful new transfer from this audit was measurement-driven placement of
the small repeatedly-read MTP sidecar, plus the missing productivity and
overlap instrumentation. FreeToken itself has target-verification kernel
hooks, but no end-to-end speculative decoder in the reviewed revision.

Sources: https://github.com/FlashML-org/FreeToken and
https://arxiv.org/abs/2608.16157 (repository revision
`bd372b630a028e3faa51f4ab0ef6a98c2f2de501`).
