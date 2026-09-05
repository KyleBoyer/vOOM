# GLM-5.3-Flash on the 16 GB M4

## Exact packed routed-expert QMV (2026-09-04)

`VMODEL_GLM53_FP8_DIRECT_QMV=1` is a new explicit lossless representation and
kernel path for both attested GLM-5.3 targets. Routed-expert pages remain in the
checkpoint's released E4M3 bytes plus FP32 128x128 multipliers while cached. A
one-position BF16 projection uses a Metal QMV that mirrors MLX's singleton BF16
GEMV reduction partition; wider shapes reconstruct the existing exact BF16
carrier and call the ordinary matmul. Trunk, shared-expert, embedding, and head
weights stay on their existing paths.

| target / prompt | control wall | packed-QMV wall | prefill | decode | peak Metal | exactness |
|---|---:|---:|---:|---:|---:|---|
| Flash, 28 in / 8 out | 123.311 s | **115.600 s (-6.25%)** | 68.932 -> 67.405 s | 54.371 -> **48.189 s** | 3.276 -> 3.087 GB | tokens + 159/159 tensors |
| Flash coding, 12 in / 4 out | 81.331 s | **73.199 s (-10.00%)** | 41.748 -> 40.543 s | 39.575 -> **32.650 s** | 3.030 -> 2.860 GB | tokens + 159/159 tensors |
| Full, 28 in / 4 out | 362.018 s | **295.689 s (-18.32%)** | 283.101 -> **235.301 s** | 78.917 -> **60.387 s** | 6.426 -> 6.022 GB | tokens + 79/79 tensors |

The coding trace recorded 9,933 direct calls and 2,355 wider fallbacks; the full
trace recorded 16,452 and 10,095 respectively. Per-request telemetry also
separates fallback BF16 reconstruction seconds/bytes from ordinary fetch-time
FP8 transformation, preventing the packed path from hiding that remaining
cost. The full trace had one 20.1MB pressure-growth event. Consequently both
profiles remain opt-in: `glm53-flash-lossless-native-mtp3-workers2-direct-fp8-qmv`
and `glm53-full-lossless-direct-fp8-qmv`.

An exact width-one gate/up fusion was also tested and rejected (0.8523 ->
0.8565ms, plus 33.6MB output). The direct kernel therefore stays projection
granular; there is no retained fusion switch. A compact-page scheduling sweep
also retained batch 8 / depth 2. Depth 3 was neutral/slower (73.298 -> 73.360s).
Batch 16 cut expert-prefetch wait by 12.169s but made exact wider reconstruction
5.313s slower under I/O/Metal contention, regressed total wall to 75.327s, and
raised peak Metal by 227MB; its temporary validator expansion was reverted.
Applying the singleton QMV independently to wider rows was not exact against
MLX's wider GEMM at any tested width from 2 through 16, so that tempting
shortcut was also rejected before a runtime switch was added.

Status date: 2026-09-04

> Current-tree correction: `models/GLM-5.3-Flash` is the attested
> `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` revision
> `d21b19569d30e6f471c433b11e672b3bbb80552a`, retaining vision and native MTP.
> The source remains on Workspace NVMe.  A fresh exact depth-three/four A/B
> found depth four 2.40% faster on one accepted-proposal trace but 0.96% slower
> on a zero-acceptance coding trace; all tokens and 159 endpoint tensors matched.
> Depth three therefore remains the general explicit profile.
>
> The current strongest sustained-output composition is
> `glm53-flash-e-compact-mla-tile128-workers2-direct-fp8-qmv-mtp3`. On the
> 2,123-input/32-output semantic gate it preserved the exact output hash,
> accepted 22/24 native-MTP proposals, and reduced wall 784.748 -> 428.803s
> versus the original compact control. It remains explicit/E-class because its
> compact absorbed-MLA parent reassociates floating-point operations; Plex,
> varied-shape, streaming/sampling, long-context, and vision promotion gates
> remain open.

## Checkpoint and storage

- Current checkpoint: `dealignai/GLM-5.3-Flash-UNCENSORED-FP8`
- Pinned revision: `d21b19569d30e6f471c433b11e672b3bbb80552a`
- Base release: `zai-org/GLM-5.3-Flash` revision
  `04c4e9e95c5da8862dced7e5056455116f83a7e0`
- Local source: `models/GLM-5.3-Flash` on Workspace NVMe
- Released tensor payload: 328,326,771,576 bytes across 62 shards
- Hub verification: all 72 repository files passed strict cache verification
- Target representation: released fine-grained FP8 E4M3 payload with the
  released float32 128x128 inverse scales. The runtime does not add another
  quantizer.
- Exact internal mirror: 12,932,311,928 bytes under
  `~/vmodel_fast_tier/GLM-5.3-Flash`. It contains all 1,455 deterministic
  trunk/router/shared-expert tensors. Routed experts remain on Workspace.
  The published plan leaves 15.63 GB internal free and 53.93 GB total in the
  globally capped fast tier.

Old local Kimi K2.5, Kimi Linear, GLM-4.7-Flash, and remaining GLM-4 Hugging
Face cache entries were removed at the user's direction. GLM-5.3-Flash and
its 328 GB source remain on Workspace NVMe, not NAS.

## Full uncensored GLM-5.3 target

The complete attested `dealignai/GLM-5.3-UNCENSORED-FP8` revision
`aff05d054bf581b95bdfd87ba9792dbf1e4365b2` is separately local on Workspace
NVMe: 282 shards and 755,663,675,562 verified candidate bytes. Its architecture
and indexed tensor names match the official `zai-org/GLM-5.3` layout. Its 78-layer
`glm_moe_dsa` runtime now completes the unmodified 178,616-byte / 134-tool
captured harness request at 46,849 rendered input tokens. Only the model alias
and max-output-one capacity cap differ from the capture. Exact released
FP8/BF16 execution took 3,573.109 seconds wall / 3,567.451 seconds prefill,
read 740.041GB, peaked at 6.024GB Metal, and completed without retry. The
one-token response is incomplete by construction and cannot establish answer
quality. Cumulative host swap-outs grew 258.26MB, so this is a functional
capacity result rather than a pressure pass.

The full target's explicit long-context route is lossless: exact tiled DSA
top-k with chronological gather, query-range selection spill, compressed MLA
K/V spill, byte-identical absorbed attention at query width 32, and dense MLP
tiling at 512 positions. The run avoided 717,654 candidate-ID sorts and 29,421
final chronological sorts. Its selection tier made 79,857 reads but only 19
dirty-boundary flushes, writing 7.707GB and reading 20.920GB; compressed MLA
spill wrote 4.233GB. Selection scoring accounted for 481.262 seconds. All of
these controls remain explicit/default-off pending the heterogeneous corpus
and large-context conformance ladder.

Storage fetch batch eight plus one-batch prefetch is exact but does not help
the full target's non-coalesced expert loop. On the deterministic 2,219-token
real-weight gate it preserved the known output SHA but regressed wall 662.133
-> 699.684 seconds (+5.7%) and prefill 659.577 -> 695.778. It hid 85.924
seconds of I/O but waited 529.797 seconds, at 2.278GB peak Metal and 63.275MB
cumulative swap-out growth. Full-target defaults therefore remain batch one
and prefetch off.

## Released architecture implemented

The runtime validates and executes the checkpoint's nested `glm5_next` text
configuration: 45 target layers (3 dense, 42 MoE), 4096 hidden width, 288
routed experts with top-8 routing, 34 KDA layers, 11 pooled DSA layers, mHC
4/20 hyper-connections, NoPE MLA, and the appended layer-45 native MTP block.
Fine-grained FP8 pairs are joined and widened only for the unchanged target
matmul. KDA, DSA, compressed MLA, mHC, dense/MoE MLP, untied streamed head,
and MTP state all have bounded local gates.

The MTP implementation follows the released graph: target post-final-norm
hidden state feeds `hnorm`; MTP uses a plain pre-norm residual block (no mHC),
dense NoPE MLA, the released shared/routed MoE, and the shared-head norm. Every
proposal remains subject to the authoritative target verifier and correction.

The released vision tower and multimodal projector are also wired through the
Responses image protocol. Official preprocessing is byte-identical to the
checkpoint processor, the complete MLX tower agrees with the CPU reference at
cosine 0.99958676, and a real 64x64 green-image request answered `green`. That
cold gate took 137.949 seconds, of which 0.773 seconds was the vision tower;
the tower read 1,127,254,016 bytes. Video remains deliberately fail-closed.
With the explicit int8 request-local K/V candidate, the same real green-image
gate again answered `green` in 152.694 seconds versus its paired 158.680-second
BF16 expanded-K/V run. The 3.8% gain is below the 10% promotion threshold.

## Measured exact gates

All numbers below use the same real 28-token prompt and return the identical
four token IDs `[3411, 1172, 279, 21595]` (`" Return only the clause"`). The
short prompt is a bring-up/timing gate, not the captured 134-tool Plex request.

| Path | Cold wall | Exact repeat wall | Decode on repeat | Peak Metal |
|---|---:|---:|---:|---:|
| Plain target | 183.51 s | n/a | 58.45 s | 2.11 GB |
| Native MTP, original storage | 162.19 s | 33.53 s | 33.53 s | 2.18 GB hot |
| MTP + grouped/pipelined experts | 145.78 s | 31.02 s | 31.02 s | 2.60 GB hot |
| MTP + exact two-device trunk prefetch | **128.54 s** | **23.05 s** | **23.05 s** | **3.01 GB hot** |

The final hot run accepted 2/2 proposals in one target sweep. Its immutable
target+MTP prompt boundary was 148,121,192 bytes. It read 12.754 GB from the
internal exact mirror and 19.785 GB from Workspace. Forty-three trunk pages
(12.082 GB) were useful prefetches; at least 6.511 seconds of their service was
hidden and only 0.198 seconds was awaited. Exact repeat is 7.96x faster than
the plain cold four-token path and 31.3% faster than the first hot-MTP result.

The real two-position target-verifier oracle is separately exact: logits and
greedy tokens match two ordinary sequential target sweeps, all 146 checked
KDA/conv/MLA/DSA endpoints match, and compact KDA factor commit reduces the
two-position wall from 38.96 to 28.27 seconds.

A sustained max-16 repeat also preserves all 16 token IDs across cold and hot
runs. The hot adaptive-depth path took 146.57 seconds, used six target sweeps,
and accepted 9/13 proposals (69.2%) at 3.33 GB peak. This is a correctness and
rollback gate, not a latency success: it remains above 90 seconds. It exposed
and fixed one fail-closed bookkeeping bug—an ordinary width-one controller
round has no rejected KDA suffix and therefore must not demand speculative
rollback factors. A forced depth-five arm reduced sweeps to five but widened
routed unions, regressing hot wall to 177.66 seconds and reads from 201.71 to
219.90 GB; it was removed. Expert storage batch 16 likewise regressed the
four-token hot gate from 23.05 to 23.65 seconds and raised peak memory, so the
measured ceiling remains eight.

## Long-context measurements and candidates

The exact expanded-K/V prefill cache removes repeated projection of every old
compressed MLA row. At 2,121 prompt tokens it reduced exact cold prefill from
754.239 to 445.758 seconds while preserving the greedy token hash. Repeating
the same request through the immutable hybrid prefix endpoint reused
2,121/2,123 tokens and reduced suffix prefill to 15.656 seconds, a 28.5x
first-token/prefill improvement with the same output.

The selected-row Metal attention kernel is a separate, explicitly lossy
candidate because its online softmax changes floating reduction order. At the
real 8K/Q32/2,048-selected-row geometry it is 53.63x faster than the bounded
gather reference and saves 1.616GB peak, but is not BF16-byte-identical
(cosine 0.99994886). On a real 8,215-input/one-output request it completed in
1,276.755 seconds versus the exact path timing out above 1,800 seconds, with a
3.715GB peak and the same one-token hash. A hot-prefix 64-output control then
took 699.032 seconds: prefill was 17.152 seconds and target-only decode was
681.879 seconds. It retained both long-context canaries and all 64 tokens, but
produced only two of the four requested consecutive validation integers; that
quality miss is recorded as a failure, not adjusted away.

The exact/default-off DSA pool cache now uses stepped raw and derived backing
allocations and derives chronological metadata arithmetically instead of
rebuilding it for the full prefix. At the 46,849-position weights-free shape,
the old full-prefix rebuild measured 6.64--7.01 seconds versus 0.743 seconds
for the stepped path (about 9x), with byte-identical final pooled keys and
60.8MB versus 219.6MB peak. Raw packed append alone improved 1.038 -> 0.765
seconds (1.36x). The real captured Flash run reduced pool construction from
9.747 to 2.626 seconds while computing 128,843 rows, reusing 23,592,800, and
avoiding construction of 23,721,643 metadata rows. A gated 16/32/64/128
tile-width ladder remains available for the already-lossy fused path; no
candidate becomes a default from this single prompt.

Native MTP on the same 8,215-input/64-output gate completed in 1,783.844
seconds cold. Its state-only draft prefill covered 8,214 rows, it accepted
46/49 proposals (93.9%), and 17 target sweeps reduced decode from the
target-only 681.879 seconds to 544.641 seconds (-20.1%). Both canaries, the
exact required prefix, and all 64 tokens survived. The run still failed the
same four-integer quality condition and its 103.4MB swap-out growth exceeded
the pressure gate, so it is evidence for the sidecar speed lever, not a
promoted profile.

At the released 8K/pool-4/index-dim-128 geometry, the first incremental-pool
implementation preserved final pooled keys byte-for-byte, took 0.0962 versus
0.2274 seconds across 192 updates (2.36x), and reduced peak by 20.65MB. The
larger 46,849-position gate and real capture above supersede its performance
scope while retaining the same exactness requirement.

The real 2,123-input/max-1 tile ladder retained output SHA-256
`58bb119c...8909cb5` at every rung. Tile 64 reduced engine wall from the
445.758-second tile-32 result to 371.094 seconds (-16.8%); tile 128 reached
359.127 seconds (-19.4%) at a 3.293GB peak. Phase attribution at tile 128 was
290 seconds MLP/expert work, 58 seconds attention (53 KDA, 5 MLA), 4 seconds
weight wait, and roughly 2 seconds mHC bookkeeping. Swap-out grew 165MB, so
the wider tile is a latency result but has not cleared the pressure gate.

Coalescing all positions for each expert into one larger GEMM was tested and
removed from the short/default path. The first implementation improved
tile-128 wall only 1.2% (359.127 to 354.791 seconds) while useful
expert-prefetch overlap collapsed from 147.8 to 35.7 seconds and wait rose from
135.2 to 246.6 seconds. A later explicitly lossy implementation is now retained
default-off for long prompts only: one real expert's 1,465 one-row calls became
36.15x faster when coalesced, and the real 8,215-input/max-1 gate preserved its
known output hash while wall fell 1,279.311 to 693.561 seconds (-45.8%). The
same candidate regressed the 2,123-input gate by 4.2%, and the 8K run grew
swap-outs by 28.64MB, so no automatic length threshold is enabled.

An untouched 46,849-token coalesced attempt exposed why the first form could
not scale safely. After about 45 minutes, a hot expert's full-context gathered
operand produced a learned 2.5+GB transient; a 2.91GB expert-page reservation
was correctly refused at 4.15GB active / 7.33GB live ceiling. The generic
chunk retry proposed tile eight, but that tile did not bound an expert gathered
across all tiles, so the retry was stopped. The replacement keeps each expert
page resident once while splitting only its gathered rows at a configurable
position ceiling. A real layer-3/1,465-row sweep measured 34.99x at ceiling 512
versus 36.71x unbounded (95.3% of the speedup). On the repeated 8,215-input
gate, ceiling 512 preserved the established one-token hash, completed without
retry in 720.616 seconds, and reduced true peak from 3.785 to 3.673GB. It was
3.9% slower than unbounded, split 1,271 experts across 14,640 GEMMs, and grew
swap-outs by 38.14MB. This remains a default-off capacity candidate.

The subsequent committed bounded-512 run completed the untouched 46,849-token
/ 134-tool capture on its first attempt. Wall was 4,156.344 seconds versus
7,845.622 for the non-coalesced capacity baseline (-47.0%); prefill was
4,150.574 seconds and true peak was 7.259GB versus 7.524GB. MLP/expert time
fell 5,364.407 -> 1,710.744 seconds (-68.1%), while attention changed only
2,358.732 -> 2,287.530 seconds (-3.0%). The trace executed 37,803 coalesced
GEMMs, split 6,315 experts, observed maximum width 512, and never retried. The
capture's stochastic sampler was preserved, so native MTP correctly fell back.
Client swap-out growth improved 354.14 -> 177.08MB but still fails pressure;
max-1's 15-point Plex score is not an intelligence result.

The next explicit composition widened the content-blind tile to 128, enabled
storage fetch batch eight plus one-batch prefetch, and used the exact stepped
pool cache while retaining the same lossy attention/int8-KV/coalesced-expert
components. On the same captured request it completed in 2,778.144 seconds
wall / 2,772.693 seconds prefill, **33.2% faster** than the 4,156.344-second
bounded-512 control. MLP fell 1,710.744 -> 988.515 seconds, MLA attention
1,014.799 -> 472.196, pool build 9.747 -> 2.626, and selection 78.241 ->
60.362. Peak Metal increased 7.259 -> 7.584GB. Cumulative swap-outs grew
98.48MB and failed the strict 64MB pressure gate. The capture's stochastic
sampling was preserved, so its different one-token hash is not an identity
failure or an identity proof. This is a latency result for an already-rejected
lossy profile: the direct Plex gate remains capped at 79/100 with missing
required call fields.

The KDA recurrence itself now has a separate exact graph-compiled option.
`VMODEL_GLM53_COMPILED_KDA_PREFILL=1` preserves the ordinary MLX operators,
FP32 reduction order, and 32-position state materialization cadence. At the
released H64/D128/L128 geometry, output and recurrent state were byte-identical
and median scan time improved 0.07355 -> 0.06271 seconds (1.173x). On a real
2,123-token gate the greedy output SHA remained
`58bb119c...8909cb5`. Compared with the immediately preceding otherwise-
identical arm before the layer-stationary call-site fix, wall improved 351.929
-> 343.405 seconds (-2.4%) and KDA attention 57.706 -> 48.922 seconds
(-15.2%); MLP was unchanged at 281.49/281.56 seconds. Peak Metal increased
3.288 -> 3.299GB and cumulative swap-outs grew 17.924MB. The 46.8K trace spent
934.812 seconds in KDA scan, so the measured phase ratio projects about 142
seconds further improvement, but a projected 43.9-minute capture is not an
achieved timing. The switch remains explicit pending the anti-overfit corpus.

The compiled graph's segment length is independently identity-bound and may
be set to 16/32/64/128. Segment 16 reduced the synthetic released-geometry scan
peak from 850.4MB to 714.6MB, but the real checkpoint gate rejected it as an
end-to-end promotion: all 16 greedy tokens and every captured state component
were exact, while wall changed only 374.604 to 374.397 seconds, prefill grew by
0.512 seconds, and swap-out growth increased by 2.671MB. It remains an explicit
low-peak experiment only.

One-page routed-expert prefetch has a separate exact profile layered on the
compiled-KDA route. A fresh same-commit 16-output A/B preserved tokens, text,
and complete attention/DSA/recurrent/convolution/hidden state. It hid 5.877
seconds of future-page disk service, moved wall 374.604 to 368.970 seconds, and
reduced swap-out growth from 23.216MB to 14.418MB. Because the 1.50% wall delta
is close to run variance, a second deterministic 2,123-input/max-1 HTTP shape
was run. It retained the known output SHA, identical 300.530GB reads and
3.936GB Metal peak, while wall fell 753.260 to 583.754 seconds (-22.50%) and
engine time fell 750.353 to 580.818 seconds. The pipeline submitted 11,435
future reads, hid 215.573 seconds of service, and reduced swap-out growth from
50.348MB to 44.581MB. It is therefore the preferred explicit lossless prefill
profile, but remains non-default until real tool/harness and Plex shapes pass.

The next explicit exact rung groups eight expert pages per storage fetch while
keeping `expert_compute_batch=1`. This can amortize storage-call overhead
without changing released matmuls or ascending accumulation order; the live
governor may only clamp the storage group downward. It is isolated as
`glm53-flash-lossless-expert-prefetch-batch8` pending real gates.

Native FP8 reconstruction now has a thread-attributed hybrid experiment. The
2,123-input trace showed that running exact fused reconstruction inside both
expert-prefetch workers competed with the foreground MLP: reconstruction fell
116.343 -> 52.828 seconds, but hidden prefetch fell 300.396 -> 240.820 seconds
and MLP rose 345.334 -> 373.704 seconds. The explicit
`glm53-flash-lossless-expert-prefetch-batch8-workers2-native-fp8-foreground`
profile therefore retains fused reconstruction only outside
`vmodel-expert-batch`; background pages use the exact eager decoder.

Two fresh-process hybrid runs were nearly identical at 426.610/426.630 seconds
engine and 429.439/429.504 seconds HTTP wall. They preserved the established
output SHA, exact 300.933GB read count, and 3.936GB Metal peak. Background
instrumentation reported 34,353/34,353 eager transforms and 179 foreground
native transforms. This is 6.81% faster than the rejected all-native medium
arm, but only 0.49% faster than the two eager controls' engine mean, so it is a
confirmed scheduling mechanism and explicit medium experiment—not an
automatic threshold or replacement for the all-native short/vision leader.

The native Metal KDA recurrence also has an isolated lossy research profile,
`glm53-flash-lossy-native-kda-isolated`.  Unlike the earlier composite fast
profiles, it begins with the released-weight exact expert-prefetch route and
changes only KDA prefill.  On the deterministic 2,123-input/max-1 request it
finished in 400.823 seconds engine / 403.616 seconds HTTP wall, 6.51% / 6.53%
faster than the two eager exact controls' means.  KDA fell from a 47.247-second
mean to 21.091 seconds and peak Metal stayed at 3.936GB.

The matching output token is not an exactness proof.  Five expert routes
changed, exact weight traffic grew by 125,859,840 bytes, and the low-level
released-geometry oracle has nonzero error.  Four-output and varied six-output
gates kept every greedy target token but changed the persisted state; they
also showed no meaningful short-context speedup at 74.678 and 92.993 seconds.
Keep this profile default-off and call it lossy.  It needs a real Plex tool
quality gate and a longer-output gate before it can become a recommended fast
profile.

Those real gates now pass. On the 16-output full-state gate, batch 8 preserved
all tokens, text, aggregate state, and each attention/DSA/recurrent/convolution/
hidden hash while improving wall 368.970 to 351.396 seconds (-4.76%). Peak rose
167.6MB and swap-out growth rose by 14.0MB but remained below the 32MB gate.
On the deterministic 2,123-input/max-1 HTTP gate, request/output hashes and the
300.530GB read count remained identical. Wall fell 583.754 to 441.689 seconds
(-24.34%) versus batch 1 and 753.260 to 441.689 seconds (-41.36%) versus no
prefetch. Grouped submissions fell 11,435 to 1,448, hidden service rose to
256.396 seconds, wait fell to 31.950 seconds, peak changed by only 16KB, and
swap-out growth improved by 1.77MB. Batch 8 is the preferred explicit lossless
short/medium-context prefill profile. The untouched 46,849-token/134-tool
capture rejected it before layer 3's attention tile completed at token 24,160:
observed Metal reached 8,501,319,252 bytes against the hard 8.5GB cap and
triggered fail-fast retry after 730.041 seconds. Post-failure availability was
3.32GB and swap-out growth was 34.75MB. It is therefore not a long-context
profile. Batch 1 progressed farther but also failed in layer 3 at token 28,544
after 889.812 seconds, with 8,501,617,500 observed bytes against the 8.5GB cap,
3.88GB post-failure availability, and 39.80MB swap-out growth. The upgraded
artifact captured the sanitized reason, subphase, layer, token boundary,
observed/limit bytes, and retry chunk inline. All exact expert-prefetch widths
are therefore short/medium-context only; long context disables prefetch. Plex
remains pending.

`glm53-flash-e-compact-mla` is the next explicit memory-sidequest candidate.
The absorbed-MLA flag previously built a full expanded K/V cache before
ignoring it. V1 removed the cache globally and was rejected: all 16 short-gate
tokens and every state component changed for only a 374.604 -> 373.779-second
wall change (-0.22%). V2 keeps ordinary expanded dense attention through the
released 2,048-key boundary, then releases that layer's dead expanded prefix
and reads compact latent rows only when DSA selection is active. The sparse
formulation still reassociates floating-point products, so this remains
E-class/lossy. Its next gates isolate the lever with expert prefetch, fused
sparse attention, int8 K/V, and coalesced expert positions off.

The same storage grouping has a separate full GLM-5.3 composition with exact
DSA index preallocation. On a deterministic 2,123-input/max-1 released-model
gate, request/output hashes and all 708.060GB of reads matched the conservative
control. Wall fell 826.026 to 695.088 seconds (-15.85%), index grows fell 63 to
21, and copied index rows fell 64,512 to zero. Prefetch submitted 2,323 grouped
reads and hid 83.225 seconds. Peak rose from 1.070GB to 2.246GB but remained
far below the 8.5GB ceiling; swap-out growth improved from 75.006MB to 49.070MB.
`glm53-full-lossless-preallocate-prefetch-batch8` is therefore the preferred
explicit short/medium-context full-model schedule. The real 46,849-token /
134-tool captured replay rejected it after layer 7: 4.70GB active plus the
next 2.50GB attention reservation and 0.40GB margin projected 7.60GB against a
7.58GB live ceiling. The runtime correctly triggered a tile-8 replay, which is
already known to be catastrophically slow, so the run was stopped. A batch-4
profile halved the future storage group and cleared the early boundary, but a
fail-fast capture rejected it at layer 37 after 1,562.664 seconds. Active memory
was 4.48GB and the next 2.59GB attention reservation plus 0.40GB margin
projected 7.48GB against a 7.31GB ceiling. The recorded `memory_retry 1/4`
stopped the request before replay. Batch 2 is the next pressure-reduced rung.
Batch 2 then cleared layer 37 but failed at layer 49 after 1,991.985 seconds:
4.29GB active plus the next 2.59GB reservation and 0.40GB margin projected
7.28GB against a 7.25GB ceiling. Batch 1 cleared those boundaries but failed
after layer 75 at 3,119.328 seconds: 4.38GB active plus the next 2.48GB
attention reservation and 0.40GB margin projected 7.25GB against the 7.24GB
live ceiling. Its artifact records 4.07GB available and 141.84MB physical
swap-out growth. That exhausts the exact prefetch-width ladder for this shape;
long context returns to preallocation with prefetch off.

Halving the full model's outer layer-stationary tile from 32 to 16 is also
rejected as a lossless capacity technique. It completed the deterministic
2,123-input gate without retry in 715.243 seconds engine / 719.205 seconds wall
and lowered peak Metal from 1.070GB to 0.967GB, but changed the exact output
SHA (`d12fe7f6...68f7c4` -> `706e47ac...f33a4`) and total weight reads. The
outer tile therefore changes batched arithmetic and downstream routing. The
briefly-created lossless profile was removed and will not be used for the
46.8K conformance run.

The same trace identified a host-side governor cost: 626 shrink steps released
zero cache bytes. Named reversible reservations were repeatedly lowering an
already-empty cache limit, clearing MLX, then restoring the ineffective cut.
The governor now exits that shrink loop at the first zero-release step and
uses its existing bounded settle/refusal samples, clearing allocator cache
during those samples. The Metal/system ceilings and fail-closed refusal rule
are unchanged.

The untouched captured request has now completed one cold 46,849-input/max-1
run with all 134 tools and capture-derived streaming/sampling intact. Only the
model alias and explicit one-token output cap differed. Wall was 7,845.622
seconds, prefill 7,839.678 seconds, peak Metal 7.524GB, and swap-out growth
354.14MB. Phase timing assigns 5,364.407 seconds to MLP/expert work and
2,358.732 seconds to attention, versus 1.351 seconds waiting for layer weights.
The response completion proves capacity but the pressure gate fails; max-1 is
not a Plex intelligence score.

The focused Plex fixture uses the captured intent but substitutes only one
full real Plex schema, forces non-streaming/temperature zero, allows four turns,
and caps each turn at 128 tokens. Under that explicitly modified scope, BF16
expanded K/V scored 66.25/100 in 3,260.091 seconds. Int8 expanded K/V scored
79/100 but regressed to 4,432.907 seconds, omitted the required rating operator,
listed rejected titles in the visible answer, and truncated. It is rejected as
a quality/speed profile. The generic deterministic Plex policy renderer's
100/100 score is not counted as model intelligence.

## Latest verifier and expert-kernel instrumentation

The full target's released layer-78 native MTP is now executable only for a
DSA-elided context bound no larger than the released index top-k of 2,048.
The deterministic four-output oracle matched tokens, text, the final hidden
row, and all 78 MLA endpoint tensors, but zero accepted proposals regressed
wall from 362.018 to 411.465 seconds and reads by 12.36%. A minimum-margin-one
controller reduced the regression to 7.70% but remained slower and failed the
pressure gate. `glm53-full-lossless-native-mtp3-bounded` is consequently an
explicit correctness profile, not a recommended speed profile. Diagnostic
state gates can now emit privacy-safe hashes for each logical DSA row, and a
forced-rejection tiny-model oracle proves exact row rollback.

Flash layer-stationary execution now reports exact routed-expert shape
histograms. On the matched eight-output MTP gate, 5,198/7,081 calls had one
row. Shape-specialized compilation of the unchanged SwiGLU was state-exact,
but 6,527 compiled calls regressed wall 123.237 -> 210.600 seconds (+70.9%).
That execution path was removed; the arithmetic-neutral counters remain to
guide future fused kernels. BF16 `gather_mm` is not a lossless substitute
because its output failed the byte gate.

## Explicit controls

All new narrow speed paths remain opt-in pending the heterogeneous real-request
corpus required by the anti-overfit policy:

- `VMODEL_GLM53_MTP=1`
- `VMODEL_GLM53_MTP_DEPTH=1..5` (measured at 3)
- `VMODEL_GLM53_MTP_MAX_PROMPT_TOKENS=1..65536`
- `VMODEL_GLM53_MTP_CONFIDENCE_TELEMETRY=1` records content-blind top-two
  native-MTP logit margins and separates proposal attempts from discarded
  state-synchronization predictions.
- `VMODEL_GLM53_MTP_MIN_LOGIT_MARGIN=<nonnegative>` withholds a weaker native
  candidate after its exact MTP state update but before widening the target
  verifier. Threshold `1.0` is exposed only by the explicit
  `glm53-flash-lossless-native-mtp3-confidence1-workers2` experiment; it won
  two different real-weight direct-engine shapes but is not a monotonic
  acceptance oracle and is not an automatic default.
- `VMODEL_GLM53_SPARSE_FUSED_ATTENTION=1` enables the explicitly lossy
  online-softmax selected-row kernel.
- `VMODEL_GLM53_SPARSE_FUSED_KV_INT8=1` halves the expanded request-local
  prompt K/V for that already-lossy fused kernel; it requires the fused kernel.
- `VMODEL_GLM53_COALESCED_EXPERT_POSITIONS=1` uses one GEMM outer shape per
  expert across layer-stationary tiles. It is lossy, regresses the measured
  short gate, and remains a long-context experiment.
- `VMODEL_GLM53_COALESCED_EXPERT_MAX_POSITIONS=128|256|512|1024|2048|4096`
  bounds the gathered operand while retaining the expert page; default 512.
  A prefill memory retry lowers this ceiling before shrinking tile width.
- `VMODEL_GLM53_INCREMENTAL_DSA_POOL=1` enables the candidate exact immutable
  pool-key cache with stepped raw/derived capacity.
- `VMODEL_GLM53_COMPILED_KDA_PREFILL=1` selects the exact bounded graph-
  compiled KDA recurrence; it retains the reference MLX operators and
  32-position state boundaries. The real 2,123-token A/B improved wall 2.4%
  and KDA attention 15.2%, but the switch remains explicit pending broader
  request-shape proof.
- `VMODEL_GLM53_PREFILL_TILE_WIDTH=16|32|64|128` selects a gated,
  content-blind layer-stationary tile (default 32).
- `VMODEL_GLM53_EXPERT_FETCH_BATCH=1..8` (measured at 8; 16 regressed)
- `VMODEL_GLM53_EXPERT_BATCH_PREFETCH=1`
- `VMODEL_GLM53_TRUNK_PREFETCH_DEPTH=0..2` (measured at 1)
- `VMODEL_GLM53_TRUNK_PREFETCH_WORKERS=1..2` (measured at 1)
- `VMODEL_GLM53_NATIVE_FP8_DEQUANT=1` fuses the released E4M3 decode,
  arbitrary FP32 128x128 block multiplier, and BF16 rounding into one Metal
  dispatch. It is byte-exact against the eager decoder over all payloads and
  real checkpoint tensors; current full/Flash end-to-end wins are 3.29% and
  6.72--7.65%. The independent green-image gate also preserved `green` while
  improving wall 103.891 -> 98.958 seconds and reducing peak Metal 2.869 ->
  2.745GB. A deterministic 2,123-input Flash A/B/A then rejected this switch
  with the exact two-reader profile: eager controls took 433.005/430.597s and
  native took 460.790s despite identical output/read/peak evidence. Keep it
  short/vision-only pending a better overlap schedule. It does not use MLX's
  incompatible E8M0 MXFP8 matmul path.
- `VMODEL_GLM53_HOT_PROMPT_KV=1` enables the ordinary target's generic exact
  hot-prefix cache. Native MTP has its own exact paired target/draft boundary.
- `VMODEL_EXECUTION_PROFILE=layers|ops` enables bounded request attribution.

Defaults preserve the original conservative GLM-5.3 profile: expert fetch
batch 1, both prefetch paths off, native MTP off, and generic hot prompt KV off.

The full 78-layer target additionally exposes the explicit exact long-context
controls `VMODEL_GLM_DSA_LONG_CONTEXT=1`,
`VMODEL_GLM_DSA_SPARSE_ABSORBED_MLA=1`,
`VMODEL_GLM_DSA_MLA_KV_SPILL_DIR=<external-volume-path>`, and bounded key,
query, index, and dense-MLP tile settings. Its storage-only expert grouping
reuses `VMODEL_GLM53_EXPERT_FETCH_BATCH` and
`VMODEL_GLM53_EXPERT_BATCH_PREFETCH`; arithmetic still executes one expert at
a time in released ascending order. Batch-eight/two-reader expert prefetch is
now the measured full-model base; an 850MB exact trunk page cache, one-page
trunk prefetch, a checkpoint-bound 14.300GB internal raw tier, and native FP8
reconstruction reduce the same 17-input/max-1 path from 149.189s to 131.684s
while preserving output and the exact 214.904GB read total. The composition is
still explicit and is not valid as a workaround for the known long-context
expert-prefetch residency rejection.

## Scope still to prove

- The unmodified captured request renders to 46,849 input tokens and all 134
  real tools. Full uncensored GLM-5.3 now completes its lossless one-token
  capacity gate in 59.46 minutes of prefill, while the explicit lossy Flash
  composition takes 46.21 minutes. Both fail pressure; a complete tool/Plex
  workflow is still outstanding and cannot be inferred from max-1.
- Native MTP now has an explicit 65,536-token ceiling and a state-only draft
  prompt prefill that computes exactly the compressed latent retained by
  draft decode, skipping dead prompt Q/O/MLP work. The 8K/64-token real gate is
  the first long admission test; larger context-ladder rungs remain pending.
- The 32K -> 128K -> 256K -> 512K -> 1M context ladder remains outstanding.
- No new GLM-5.3 behavior becomes automatic until token/hash/state gates pass a
  heterogeneous corpus spanning tool counts, system/developer shapes,
  streaming modes, sampling modes, and output lengths.
