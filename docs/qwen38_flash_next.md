# Qwen3.8-Flash-Next local bring-up

Status: the pinned checkpoint download and checksum gate completed on
2026-08-27. Exact released-BF16 text serving, the PLE direct-row provider,
hybrid prompt-state persistence, and target-authoritative Lightning-MTP are
implemented and real-checkpoint tested. On 2026-08-28 the unchanged 49,255-
token harness cleared the sub-90-second max-16 goal at 76.0569s after a durable
exact endpoint restart. Optimization candidates remain profile-scoped until
their memory and heterogeneous-replay gates pass.

## Source and storage

- Source: `Qwen/Qwen3.8-Flash-Next`
- Pinned revision: `f5d08274bafd880402bd16f5e3e6c514136ec06c`
- Destination: `models/Qwen3.8-Flash-Next`
- Downloaded: 144 remote files, including 131 safetensor shards; 360.0GB
  decimal / 335GiB of tensor payload and metadata
- Verification: `hf cache verify` matched all 144 remote checksums against the
  pinned revision. Its warning names exactly 291 local files, all under the
  Hugging Face client's `.cache/huggingface` bookkeeping tree; there are no
  unaccounted model files outside that tree.
- Post-download free space: 115GiB on the workspace NVMe and 34GiB on the
  internal root volume
- Space reclamation: the exact local directory
  `models/Qwen3-VL-235B-A22B-Instruct` (461,525,492KiB) was removed after the
  runbook identified the model as NAS-archived. The directory was not a
  symlink and no process was using it. This removal is not locally recoverable.

The checkpoint must stay on the external project NVMe. Do not put its full
weights or PLE table in `~/vmodel_fast_tier`; that tier remains globally capped
at 90GB with at least 10GB actually free.

## Released architecture

The released config identifies `Qwen4ExpForConditionalGeneration` /
`qwen4_exp`: 48 layers arranged as 12 repetitions of three Gated DeltaNet
layers and one Qwen Sparse Attention layer. The text model has hidden size
2,560, 512 routed experts with top-10 routing, intermediate size 640, a shared
expert, four gated-residual streams, native 262,144-token context, and one
multi-step MTP layer. Qwen reports 125B main-model parameters with 6B active,
plus 51B n-gram-embedding parameters and 4B MTP parameters.

The PLE/n-gram table is the unusually favorable part for this Mac. It is used
at one early layer, hashes 2- and 3-token context into 16 row IDs, and returns
16 × 160 BF16 values per token (5,120 bytes). The released table contains 128
numbered BF16 tensors with shape `[2,500,012, 160]`, totaling 102,400,491,520
bytes of row payload plus 65,679,640 bytes of related buffers. It spans 33
safetensor files. The roughly 95.43GiB table must never be loaded or copied
into Metal. It is now served by authenticated sorted/coalesced direct row
reads while active MoE expert pages can stream independently.

## Measured checkpoint inventory

Header-only inspection found 1,658 tensors across 131 shards and
359,999,963,128 tensor bytes. Every floating tensor is BF16; the only other
tensors are three I64 PLE buffers. No unknown dtype was accepted.

| Category | Released bytes |
| --- | ---: |
| Routed experts | 241,591,910,400 |
| PLE / n-gram | 102,466,171,160 |
| MTP | 5,214,301,696 |
| Linear attention | 4,173,020,928 |
| Gated residual | 1,281,249,280 |
| Full/sparse attention | 1,195,388,928 |
| Token embedding | 1,271,398,400 |
| LM head | 1,271,398,400 |
| Vision | 897,862,112 |
| Shared experts | 472,104,960 |
| Router gates | 125,829,120 |
| QSA indexers | 39,327,744 |

At one generated position, ten cold routed experts per layer account for
4,718,592,000 bytes / 4.395GiB across the complete 48-layer target sweep
(about 98.3MB per layer), before trunk, shared-expert, and head reads. That is
the first major streaming and placement target. The 5.214GB MTP block is too large to make
resident on this 16GB machine; it must remain opt-in and prove that avoided
target sweeps repay its I/O.

Primary references:

- https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- https://github.com/QwenLM/Qwen3.8-Flash-Next
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modular_qwen4_exp.py

## Implementation and proof order

1. **Complete.** Verify all downloaded files and inventory every safetensor header. Report
   actual bytes by PLE, routed experts, shared/dense trunk, QSA/indexer, GDN,
   vision, LM head, and MTP. Confirm the real PLE tensor split/layout before
   writing a row provider.
2. **Complete.** Add config-only `qwen4_exp` parsing and a header oracle. This stage must not
   import MLX or load a tensor.
3. **Complete for the storage/address layer.** Implement exact PLE ID generation and direct row paging. Compare IDs and
   BF16 row bytes against the pinned Transformers implementation over EOS,
   chunk boundaries, cache continuation, and randomized token histories.
   Authenticate source blocks before exposing rows to MLX; do not materialize
   a duplicate 95GiB sidecar. The live test requests 48 rows and proves only
   15,360 payload bytes are read. MLX integration remains part of text-model
   bring-up.
4. Adapt the existing Qwen Gated DeltaNet recurrence with the released
   16-key/48-value-head geometry. Released operator order, FP32 recurrent
   state, convolution history, and arbitrary split continuation must match the
   reference before any fused/chunked candidate is admitted.
5. Implement gated residual and sparse MoE. Stream only the ten selected routed
   experts plus the shared expert. Prove router IDs/weights and layer outputs
   against the official implementation, then measure two-device placement
   from actual trace-weighted bytes rather than dividing files by size.
6. Implement QSA indexer and micro-block attention. Require selected block IDs,
   gathered KV, logits, output, and cache continuation to match the released
   eager reference. Any reassociated Metal softmax remains experimental until
   greedy byte identity clears a heterogeneous corpus.
7. Bring up text-only greedy generation before vision. Add the built-in MTP
   only after plain autoregressive state and token oracles pass. Speculative
   verification must remain target-authoritative and cover every accepted
   prefix, rejection, EOS/stop, and long recurrent rollback.

Every live rung requires a fresh 30-second memory preflight, one Metal job at a
time, peak Metal at or below 8.5GB, at least 10GB workspace free, and exact
released-model tokens against the pinned reference. New routing or lossy paths
remain explicit opt-ins until the multi-shape replay corpus passes.

## Untouched 49K agent capture: measured optimization decisions

The current real-traffic gate is the byte-for-byte captured 134-tool request
`logs/captured_requests/1784574315421_94161f5f.json` (SHA-256 prefix
`8ac18b8e`). It renders to 49,255 tokens without replacing tools, messages,
temperature, or streaming mode. The table below uses repeat 2 from the same
server process where available, because a fresh Qwen4 engine adds about 15
seconds of initialization that is not decode work.

| Exact target candidate | Warm wall | Decode | Target sweeps | Proposed / accepted | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Native MTP depth 3, pipeline off | 117.935s | 99.795s | 6 | 18 / 9 | matched control |
| Native MTP depth 3, expert-page pipeline | 115.551s | 97.457s | 6 | 18 / 9 | keep opt-in; 2.0% wall gain, below 10% promotion gate |
| Native MTP depth 5 | 142.649s | 124.543s | 6 | 29 / 10 | STOP; no sweep removed and 176.0GB streamed |
| Depth 3 + exact BF16 GEMV + exact disk endpoint | **76.057s** | **73.307s** | 6 | 18 / 9 | scoped promotion; full 49,255-token prompt hit, peak 5.729GB |

The pipeline/control output SHA-256 is identical
(`411b96d6...bf2fd85`). Both repeat-2 runs violated the memory-pressure gate,
so neither is eligible as a default. The max-16 Plex score is 15/100 because
the response is deliberately truncated before it can complete the task; it is
a latency/instrumentation rung, not a model-quality score.

The exact endpoint row uses the 20.499GB trace-balanced two-device mirror, a
400MB scan cache with one-position prefetch, released Lightning-MTP depth
three, exact BF16 serial-verifier GEMV, and phase-scoped head suspension.  The
journal additionally authenticates the Qwen4 hyper-connection hidden carrier;
an older endpoint without it is not eligible for a zero-sweep hit.  The first
migration request rebuilt the five-token suffix, while the steady request
loaded all 49,255 prompt positions and preserved the established output hash.

At max-64, the same untouched request completed its output budget in 298.2957s
versus the earlier 501.1384s. It accepted 41/68 proposals in 23 target sweeps
(40 fewer than plain autoregression), peaked at 4.248GB Metal, and stayed under
the 16MB swap-out-growth gate. Its 15/100 Plex result is again an incomplete
output-cap result, not evidence of poor completed-answer quality.

Other bounded exact candidates were stopped before promotion:

- Splitting decode pages across the two independent SSDs improved a
  fresh-process sample only 3.4%, a comparison confounded by initialization;
  its memory gate failed. Physical placement remains trace-balanced rather
  than a blind half-and-half split.
- Prompt lookup found one seven-token repeated suffix in the real capture, but
  its first proposal was rejected and added a target sweep: 137.786s warm.
- Zstd level-1 over released BF16 expert pages decoded at 1.41--1.53GB/s,
  below the measured 1.62GB/s raw NVMe floor. Byte shuffle improved ratio but
  reduced end-to-end decode/unshuffle throughput to 0.73--0.76GB/s. CPU
  decompression is therefore a STOP on this machine.

The instrumented profile now reports verifier page admission, weight wait,
reserve time, linear/full-attention compute, head time, exact expert-pipeline
submission/wait/hidden time, and per-layer counts. For stochastic traffic it
can also measure expected p/q overlap for several draft-temperature scales on
the observed native-draft path. That measurement never changes the sampled
proposal or the authoritative target distribution; selection and promotion
still require a heterogeneous held-out replay corpus.
