# Memory model

Updated: 2026-07-16. Implemented/proposed labels are self-contained here; the
older references to a separate future-techniques ledger predated this document's
consolidation and no longer describe a repository file.

## Tiers (implemented unless marked proposed)

```
Tier 0/1  materialized mx.arrays in unified memory
          - pinned pages (embeddings, final norm, optional lm_head, first/last N layers)
          - WeightCache pages (LFU-admit, byte budget, optionally quantized on load)
          - KV cache resident pages + open tails
          - transient activations
Tier 1.25 (EXPERIMENTAL, off by default) exact-compressed application cache
          - admission-controlled vpack bodies in ordinary RAM
          - expands one page at a time into a bounded Metal scratch pool
          - counts against total RSS; it is not free memory
Tier 1.5  macOS compressed memory / swap (hidden, involuntary: Metal working sets
          beyond the recommended working set can be compressed or paged out —
          "resident" weights silently become millisecond-cost faults). The live
          governor uses MLX's device limit plus sampled system availability.
Tier 2a   macOS file page cache (clean pages of safetensors files; free re-reads,
          invisible to the runtime, evicted by OS pressure)
Tier 2b   disk hierarchy (N drives, fastest first — WeightStore.fast_dirs overlay)
          - staged packed layers on faster drives (internal NVMe, TB enclosures):
            bytes served here leave the slow drive's per-token critical path
          - primary model store on the big/slow drive (read-only, full precision,
            optionally vpack byte-plane compressed: ~1.34x fewer bytes, bit-exact)
          - KV spill pages (read/write, .kv_spill/*.safetensors)
Tier 3    NAS over SMB (~110 MB/s), implemented as archive/correctness tier

Split placement math: with overlapped reads, per-token time ≈ max over drives of
(bytes_on_drive / drive_throughput). Copy-ahead staging through a fast drive does
NOT raise steady-state throughput (every byte still crosses the slow drive once
per token); only *static residency* on a faster tier removes bytes from the
bottleneck. A serial fetch path must never promote a page onto a slower tier.
`runtime.storage_tiers.plan_static_placement` makes that exclusion explicit; its
parallel projection uses longest-processing-time/minimax scheduling and is not a
speed claim until the serving reader overlaps the devices. The vpack2 reader now
has that exact cross-device path for a mixed internal-overlay/archive fetch:
filesystem plus zstd/numpy work overlaps, all MLX materialization stays on the
engine thread, and `st_dev` equality forces serial fallback. A real Qwen3.6
expert-page A/B measured 0.3220 s serial versus 0.1835 s parallel median (1.755x)
over disjoint ~430 MB requests. RAM pinning (pin_first/last_layers) is the same idea
one tier up — a rotating LRU window is worthless under cyclic dense-layer sweeps
(a page is always evicted before reuse), but statically pinned layers never touch
disk. Routed-expert traces are not a pure dense sweep: some pages are genuinely
hot, but unconditional global-LRU admission can still let hundreds of one-use
pages scan those hot pages out. The implemented LFU-admit cache therefore separates
transient misses from an admission-controlled resident set.
```

Apple Silicon note: there is no CPU↔GPU copy or placement decision — unified memory
means materialized MLX arrays share physical memory with the CPU. The proposed
compressed heap is still a useful representation tier: it fits more exact pages
per byte and is not a Metal allocation, but it still consumes unified RAM and can
trigger pressure. `mx.get_active_memory()` tracks the Metal side; psutil RSS tracks
the rest; telemetry must log both.

## Page types

**WeightPage** (`runtime/weight_cache.py`) — the tensors of one transformer block,
or a pinned group (`persistent` = embeddings + final norm + optional lm_head).
Loaded via lazy `mx.load` + `mx.eval` of exactly the named tensors. Eviction =
dropping references + `mx.clear_cache()`. Pages carry an `origin` tag
(pin/demand/prefetch) used by both stats and eviction preference.

**Expert WeightPage** (implemented) — the three tensors of one routed expert,
keyed `layer.N.expert.E`. A layer's selected experts are batch-fetched after the
router runs. Shared experts currently remain in the always-needed layer page.

**Expert live-set invariant** (F74-v2 required, not yet proven) — storage-call
batching alone is not a lifetime bound. `WeightCache.get_many()` accumulates all
pages requested by one call in its returned result; under F74-v1 that call covered
the whole union, so caller/lazy-graph references survived cache eviction. Thus, for
expert page bytes `W_e`, cache bytes
`C`, compute scratch `T`, and routed union `U`, the current peak can approach

```text
P_v1 >= base + U * W_e + T
```

even when each `_fetch()` call contains only q experts. The required runner-level
invariant is

```text
P_v2 <= base + C + q * W_e + T + output_tile + declared_prefetch_slack,
live_expert_pages_not_in_C <= q
```

Each batch must be fetched, computed in released order, materialized, and released
before the next batch. The latest candidate wires this schedule from
`StreamingEngine._iter_expert_batches()` through GLM and reserves per batch. Its
pure-Python weak-reference test establishes the consumer deletion rule, but cache
accounting still cannot prove engine/MLX array lifetime; add instrumented engine
liveness and true Metal peak. For GLM only, engine construction maps a zero/unset
`expert_fetch_batch` to conservative q=1 in every construction path until larger
batches pass those gates; other MoE architectures retain zero-as-unbounded
semantics.

**CompressedWeightPage** (implemented experiment, F04; off by default) — an exact vpack body retained in a
bounded ordinary-RAM warm cache. A hit avoids disk but still requires decoding to
a reusable materialized scratch page. Synchronous recompression reduced disk reads
but increased wall 54%; async/direct encoded-body admission is the only queued revisit.

**EmbeddingRowGroup / LogitBlock** (F02) — input row groups are implemented and
A/B-verified for the 1.903 GB GLM embedding. Output LogitBlock streaming is
implemented for raw safetensors only and token-gated, but its published memory
measurement used an invalid peak bracket and is reopened. Packed/vpack GLM still
needs a row-addressable exact sidecar or independently decodable extents.

**KV page** (`runtime/kv_paged.py`) — `page_positions` positions of one layer's
K and V. Closed pages spill oldest-first to safetensors files when the KV byte
budget is exceeded; attention pages them back in transiently (attention is global,
so "older context is slower" not "older context is dropped").

**Compressed MLA/DSA page** (partial F21; proposed F07/F08) — ordinary GLM
`generate()` caches native `c_kv + k_rope`; those stored latent bytes are exact,
but current decode re-expands them under a different GEMM shape and has measured
0.000244 activation drift. Thus 49x is a state-capacity result, not an L0
released-arithmetic result. F87 proposes a bitwise projection replay residual:

```text
r = bits(P_insertion(x)) XOR bits(P_canonical(x))
B_exact = 1,152 latent/RoPE bytes + compressed(r) + metadata
raw r = 64*(192+256)*2 = 57,344 bytes/token/layer
```

It is useful only if that residual is very sparse/compressible and canonical
replay is fingerprint-stable; otherwise exact expanded K/V paging is the fallback.
Exact closed paging and separate index-key blocks remain proposed. Current DSA
decode gathers selected rows from
the in-memory latent/key arrays; it does **not** yet issue grouped disk block reads.
F08 proposes grouping selected positions into reads, and IndexShare groups may
store four layers position-major so one selected range serves all four. F56 adds
safe block bounds that may avoid some key reads after F22/F33. These change layout
and I/O only, never the selected indices or cached values.

Sparse multi-position scoring must also be tiled. A dense score tensor for
`L=64, H=32, S=1M` in fp32 is about 8 GB before candidates or attention output.
For key tile P and selected count K, keep only the previous K candidates and the
current P scores per query/head:

```text
TopK(A union B) = TopK(TopK(A) union TopK(B))
working memory = O(L*H*P + L*(K+P)), not O(L*H*S)
```

The head dimension disappears after the official signed head sum; candidate
scores/IDs are per query, not per head.

This is exact only when every tile and merge uses the released fp32 scoring,
causal mask, deterministic total tie order, and absolute positions. After the
final top-k, gather in the reference order (HF eager currently needs chronological
selected positions); set equality is not an attention-output proof.

**TurboQuant/quantized MLA page** (proposed SQ21; lossy) — released 1M state is
79.872 GB `c_kv` + 9.984 GB `k_rope` + 5.376 GB exact full-layer index keys =
95.232 GB. Quantizing only `c_kv` gives 32.832 GB total at 3.5 bits or 30.336 GB
at 3 bits before scales/codebooks/padding. The mild profile leaves RoPE/index
keys BF16 because top-2,048 membership is discontinuous. F64 is a distinct strict
experiment that adds an exact XOR residual; plain TurboQuant is never a lossless
cache.

State construction is now unified: `StreamingEngine.new_kv()` is the canonical
factory used by ordinary generation, speculation, and probes, enabling compressed
MLA and attaching DSA for GLM. Long-context equivalence remains open for different
reasons: DSA rollback trim and a >top-k regression exist, but this audit found no
source-fingerprinted raw current-tree result; MTP's private state still lacks
released IndexShare/DSA semantics and is now quarantined above `index_topk` rather
than allowed to run dense; sparse `L>1` prefill and ordered sparse-output oracle
artifacts are missing; and no >2,048 end-to-end proof has passed. The
real-weight probes use tolerance/set comparisons and do not close these gaps.

**Prefix-state delta journal** (F37 v6/F60/F67) — stores exact final logits plus
ordinary or compressed MLA KV and DSA keys as immutable parent-hashed segments.
Canonical generation IDs cover the payload SHA-256 and byte count as well as
model/runtime/arithmetic identity; publication is payload-first with fsync and an
immutable manifest commit. Loads verify every selected payload, a corrupt newest
generation falls back to an older valid one, reader leases exclude concurrent GC,
and GC enforces both generation-count and reachable-byte budgets. Reconstruction
collects segment pieces and concatenates each tensor once, so read work is linear.

Extending a known prefix now writes only delta positions plus a small endpoint
checkpoint. At released GLM geometry compressed MLA plus full-layer DSA keys is
about 95 KB/token, so a 1M endpoint still needs roughly 95 GB of retained state,
but 4K checkpoints no longer imply the former ~11.6 TB cumulative full-snapshot
write. A complete endpoint larger than the configured store budget remains
ineligible because deltas cannot make an undersized store retain it. Lookup/write
also stays gated by `VMODEL_PROMPT_KV_MIN_TOKENS` (default 2,048); the measured
724-entry v5 store made a short request slower than recomputation, and v6 removes
those obsolete full snapshots once during upgrade.

**Archive generation** (partial F31) — immutable vpack generations plus an atomic
pointer reduce mutation risk. New generations now pass every extent, compressed-
body hash, and decode before the flip; files and the containing directory are
fsynced, including the temporary `CURRENT` contents, and a 14/14 random-SIGKILL
rerun is archived. Readers still have no lease, old data retires immediately, and
initial consuming builds plus deterministic power/SMB faults remain open.
Placement calculations must reserve both generations during a rewrite; quiesce
readers and retain old data until pointer durability and reader retirement close.

**Parity expert page** (proposed F53) — an XOR page plus either exact partner
reconstructs the other expert byte-for-byte. Count parity bytes, partner residency,
read/decode time, and one reconstruction scratch buffer. It is conditional coverage,
not compression, and must beat literal fast-tier placement in trace simulation.

**QTensor** (`runtime/quant.py`) — a weight quantized *on entry to the cache*
(`mx.quantize`, 4/8-bit, grouped). Disk stays full precision; residency shrinks
4–8×. Per-module policy allows attention at disk precision + MLP quantized.

## Cache behavior

- Cumulative LFU-with-admission over unpinned pages; byte budget includes pinned pages.
- Budget smaller than one layer degrades to pass-through streaming (the caller's
  reference keeps the in-flight layer alive; the cache immediately re-evicts).
- Prefetcher uses one worker for raw safetensors and two for packed stores so zstd
  decode can overlap a disk read. It may evict consumed pages to make room ahead of
  the compute wavefront, but never evicts unconsumed prefetched pages (see
  `WeightCache.would_fit`), which prevents self-thrash at tiny budgets.
- Dedup: concurrent demand/prefetch loads of the same page are collapsed via an
  in-flight table; the loser waits on an event instead of double-reading.
- F42 synchronously reserves estimated operation scratch before allocation and
  sheds pages/prefetch when needed. The F74-v2 candidate now reserves one compute
  batch rather than the union; that is valid only if engine/MLX testing proves
  fetch-compute-materialize-release actually bounds lifetime. Reactive polling and
  cache eviction do not free arrays still held by a caller/lazy graph.

Memory/byte telemetry must separate representations. `WeightStore.fetch()`'s
returned `nbytes` is the sum of logical tensor payloads requested, not an observed
disk/NAS count. Record at least logical tensor bytes, requested file extents,
decoded/materialized bytes, cache residency, process I/O, and device/network bytes
as separate fields. Label unavailable counters unavailable; never infer physical
no-overread from equality with a header-size calculation.

Implemented expert-cache policy (F03): trace replay selected scan-resistant
cumulative LFU admission against LRU-family policies and a Belady bound. A cold
page can evict itself instead of displacing a proven-hot page. Aging, ghost seeding,
and learned next-use distance remain F38/F40 experiments.

Proposed placement objective (F05), for page `p` promoted from slow tier `s` to
faster tier `f`:

```text
benefit(p,f) = expected_accesses(p) * compressed_bytes(p)
               * (1 / bandwidth_s - 1 / bandwidth_f)
               - decode_and_contention_cost(p,f)
```

Optimize under separate Metal, ordinary-RAM, internal-staging, USB-space, and
per-device bandwidth constraints. This makes always-used shared experts compete
fairly with hot routed experts and replaces coarse "first layers first" placement.

F51 removes the artificial fixed split between resident KV and experts. For
measured latency curves `L_e(m_e)` and `L_kv(m_kv)`, allocate within:

```text
m_e + m_kv + reserved_scratch <= C_live(t)
C_live(t) = min(MLX_recommended_working_set,
                active_metal(t) + max(0, system_available(t) - critical_reserve))
-dL_e/dm_e ~= -dL_kv/dm_kv  (interior optimum)
```

`C_live` is resampled: a previously observed 8.5 GB peak is evidence for that
host/load, not a portable allocation cap.

Compressed MLA latent/RoPE grows 89.856 MB per 1,000 positions across 78 layers;
the 21 full indexer layers add 5.376 MB, for 95.232 MB total. The expert-favoring
short-context optimum therefore cannot be assumed at long context. Use an offline
curve first, with a hard KV correctness/admission floor and F42 headroom.

## Access pattern

Decode sweeps layers 0..N-1 once per token. This cyclic pattern is the worst case
for both LRU and the OS page cache when the model exceeds the budget (a page is
always evicted just before its next use). That is why:

- sequential prefetch works so well (the next needed page is always known);
- partial residency gives *linear* speedup (each resident layer removes its bytes
  from the per-token disk bill);
- quantize-on-load is the big lever: it changes whether the model fits at all.

For MoE multi-position forwards, fixed weights are reused but routed-expert bytes
depend on the union of selected experts. With `E` experts, `k` selected per
position, and independent routes, the expected union for `B` positions is:

```text
U(B) = E * (1 - (1 - k/E) ** B)
```

For GLM (`E=256`, `k=8`), `U(5)=37.6`, not 8. Speculative decoding and
multi-sequence batching must therefore report committed/output tokens per physical
byte. Speculation benefits only when accepted tokens amortize the larger union;
batching benefits when fixed bytes and overlapping expert pages amortize across
sequences. At `B=64`, the independent-route estimate is about 222 experts or
16.8 GB of 75.5 MB BF16 pages, so a context limit of 2,048 does not protect a
64-position sweep. The executable lifetime checks live in
`tests/test_expert_batching.py` and `tests/test_expert_batching_mlx.py`; the
real-scale Metal-lifetime proof remains open as described above.

Historical floor on the former USB model source (~315 MB/s cold reads): streamed
fp16 7B ⇒ ~13 GB/token ⇒ ~41 s/token. The current project NVMe is ~1.56 GB/s
sequential first-touch; the attached backup USB is only ~92 MB/s and must not be
substituted into that historical calculation.
