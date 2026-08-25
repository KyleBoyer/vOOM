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
5.5 GB available/post-generation floors. A later strict durable-prefix gate
confirmed that floor on the shipped profile; 6 GB rejected a restored suffix
before allocation even though the lower floor stayed under all Metal/swap
limits.

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

## Smaller proposal sidecar: official DFlash2, retained as short-shape opt-in

FreeToken does not provide a trained next-token sidecar, so the next proposal
source was the official Qwen3.8-27B DFlash2 checkpoint.  The runtime imports
its five-layer, five-target-tap, sliding-window draft architecture and selector
behind explicit configuration.  Local affine4/3/2 group-64 artifacts occupy
1.083GB / 842.365MB / 601.847MB.  They are never verifiers: the unchanged
target verifies every position and supplies the rejection correction or bonus
token.

The 2-bit artifact improved the unmodified capped capture by 5.68% and two
runs of a different developer/two-tool shape by 7.77–16.58%, with target-output
hashes unchanged.  It did not generalize to sustained output: at a request
budget of 64, native MTP completed the function call in 201.708s with 84.2%
acceptance, while DFlash2 took 246.971s with 18.2% acceptance.  Keeping even
the 602MB draft resident alongside streamed target layers failed the hard
governor on this 16GB machine, so reload cost cannot safely be hidden.  Native
MTP remains the serving choice; DFlash2 is useful research evidence that a
smaller sidecar can win only when its target-sweep reduction exceeds its load
and proposal cost on the actual request shape.

Primary implementation sources:
https://huggingface.co/incoai/Qwen3.8-27B-DFlash2 and
https://github.com/z-lab/dflash.

## Accepted: restart-safe mixed-depth prompt state, 75.10 seconds wall

The ordinary exact hot-KV journal assumes every attention layer retains the
same token count. The `16:1024` profile deliberately does not: its first four
full-attention layers retain all 6,332 stable-boundary positions while the
remaining twelve retain 1,024 packed suffix positions. Pretending those were
uniform deltas would be incorrect.

`runtime/qwen_mixed_depth_kv_persist.py` writes one immutable stable-prefix
snapshot containing the actual local length of all 64 layers, complete K/V for
all 16 full-attention layers, and state plus convolution history for all 48
DeltaNet layers. The manifest and 464,993,106-byte safetensors payload are
content-addressed and SHA-256 checked. Model, tokenizer, runtime source,
quantization, RoPE and schedule changes fail closed through the engine KV
fingerprint. Restore is allowed only when the snapshot tokens are a strict
prefix of the new prompt; it never supplies endpoint logits and cannot rewind
or branch.

The one-time final-fingerprint seed used the unmodified capture with a
one-token output cap, because no verifier/decode sweep is needed to construct
prompt state. It finished in 82.096 seconds wall, wrote the 443 MiB on-disk
snapshot in 0.516 seconds, peaked at 3.256 GB Metal and grew swap-outs by
6.734 MB. After stopping the server and passing a fresh 30-second memory
preflight, a new process served the unmodified capture with the ordinary
16-token benchmark cap:

| Metric | Empty persistent cache | Fresh-process restored cache |
|---|---:|---:|
| Cached prompt tokens | 0 | 6,332 / 6,339 |
| Suffix prefill | 58.687 s (prior clean cold baseline) | 10.082 s |
| Decode | 61.295 s | 60.893 s |
| Engine | 119.983 s | 71.622 s |
| End-to-end wall | 123.424 s | **75.104 s** |
| Peak Metal | 3.199 GB | 2.469 GB |
| Swap-out growth | baseline arm gate | **15.581 MB** |

The restored run passed the fixture's explicit `<90s`, `hot_disk`, 5.5 GB
available-memory, 8.5 GB Metal and 16 MB swap-out gates. It kept the baseline
response SHA-256
`9b170095a170f34a248af480bab274aef28f42ebf1504196bc28cc57e5b87b81`,
accepted 8/8 released-BF16 MTP proposals and produced 1.875 target tokens per
target sweep. This is an operational restart/warm-prefix result, not a claim
that an empty-cache 16-token request is below 90 seconds.

Artifacts:

- `logs/sub90_mixed_kv_final_seed_capture1.json`
- `logs/sub90_mixed_kv_final_restart.preflight.json`
- `logs/sub90_mixed_kv_final_restart_capture16.json`

## Dual-disk and prefetch instrumentation

New request telemetry reports per-device bytes, service time, parallel-fetch
wall time and the hidden overlap lower bound. It also fixes external raw
safetensors accounting, which previously omitted those bytes. Prefetch now
reports loads, useful and wasted pages/bytes/load time, waits, and the useful
load time hidden before demand.

On the accepted capture, all 19 prefetched pages were consumed, none waited or
were wasted, and at least 1.966 seconds of load time was hidden. The current
prefetch depth of two remains the measured choice.

### Rejected: a second target-page prefetch worker

A fresh 2026-08-25 run temporarily exposed a two-worker dense-Qwen prefetch
arm and replayed the byte-identical 178,616-byte / 134-tool capture with the
existing depth-two schedule and 16-token benchmark cap.  The required
30-second preflight passed with zero swap growth and 6.72GB available.  All 64
prefill layers completed, but the first native-MTP verifier sweep failed closed
after 105.3571 seconds: the governor correctly refused a 0.21GB target page at
2.06GB active plus the unchanged 0.40GB safety margin, with only 5.57GB system
memory available.  Physical swap-outs had already grown by 7.70MB.

This is a safety regression, not a timing win.  Two workers can leave an extra
materialized page in flight while the serial verifier needs its authoritative
page.  The temporary serving knob was removed, the one-worker default remains
unchanged, and no retry with a lower memory floor is admissible.  Evidence:
`logs/memory_preflight_qwen_prefetch_workers2_capture16_20260825.json` and
`logs/huihui_prefetch_workers2_depth2_capture16_20260825.json`.

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

## Rejected: per-layer fast-tier containers and narrower endpoint schedules

The exact FreeToken-style layer-container builder copied every target tensor
byte-for-byte into 65 final per-layer files (6.999 GB). This increased the
governor's per-layer transient reserve to roughly 0.57 GB and forced the
128/32/8/1 retry ladder without completing. The active body manifest was
restored and all 65 derived files were removed, recovering 6.999 GB. The
builder remains as a reproducible STOP experiment in
`runtime/qwen_fast_tier_layerpack.py`; it does not participate in serving.

The more aggressive content-blind `8:256` schedule preserved the captured
response hash but only reduced prefill by 4.39 seconds and finished at
119.60 seconds wall while growing swap-outs by 198 MB, so it was rejected.
Task-scoped gateway context reduced the rendered execution prompt to 1,054
tokens and preserved that same hash, but completed in 142.70 seconds under
pressure and is not promoted. Neither experiment meets the memory/timing gate.

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
