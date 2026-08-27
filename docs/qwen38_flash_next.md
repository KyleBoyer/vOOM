# Qwen3.8-Flash-Next local bring-up

Status: the pinned checkpoint download and checksum gate completed on
2026-08-27. The PLE address oracle and authenticated direct-row provider are
implemented and real-checkpoint tested. This is not yet a serving claim.

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

The ten selected routed experts account for about 4.395GiB of released BF16
weights per layer if every selected expert is cold. That is the first major
streaming and placement target. The 5.214GB MTP block is too large to make
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
