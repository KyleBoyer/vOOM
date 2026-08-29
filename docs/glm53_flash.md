# GLM-5.3-Flash on the 16 GB M4

Status date: 2026-08-29

## Checkpoint and storage

- Released checkpoint: `zai-org/GLM-5.3-Flash`
- Pinned revision: `04c4e9e95c5da8862dced7e5056455116f83a7e0`
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
Face cache entries were removed at the user's direction. GLM-5.3 itself and
its 328 GB source remain on Workspace NVMe, not NAS.

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

The next exact candidate, still default-off, caches immutable completed DSA
pool keys instead of rebuilding the entire prefix for every 32-position tile.
The runtime exposes computed/reused pool rows and phase-attributed weight,
attention, mHC, and MLP time. A gated 16/32/64/128 tile-width ladder is also
available for the already-lossy fused path; neither candidate becomes a
default from a single prompt.

Native MTP on the same 8,215-input/64-output gate completed in 1,783.844
seconds cold. Its state-only draft prefill covered 8,214 rows, it accepted
46/49 proposals (93.9%), and 17 target sweeps reduced decode from the
target-only 681.879 seconds to 544.641 seconds (-20.1%). Both canaries, the
exact required prefix, and all 64 tokens survived. The run still failed the
same four-integer quality condition and its 103.4MB swap-out growth exceeded
the pressure gate, so it is evidence for the sidecar speed lever, not a
promoted profile.

At the released 8K/pool-4/index-dim-128 geometry, a weights-free incremental
pool gate preserved final pooled keys byte-for-byte, took 0.0962 versus
0.2274 seconds across 192 updates (2.36x), and reduced peak by 20.65MB. That
isolated result is promising but small relative to the full request and still
requires a real token/state A/B.

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

## Explicit controls

All new narrow speed paths remain opt-in pending the heterogeneous real-request
corpus required by the anti-overfit policy:

- `VMODEL_GLM53_MTP=1`
- `VMODEL_GLM53_MTP_DEPTH=1..5` (measured at 3)
- `VMODEL_GLM53_MTP_MAX_PROMPT_TOKENS=1..65536`
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
  pool-key cache.
- `VMODEL_GLM53_PREFILL_TILE_WIDTH=16|32|64|128` selects a gated,
  content-blind layer-stationary tile (default 32).
- `VMODEL_GLM53_EXPERT_FETCH_BATCH=1..8` (measured at 8; 16 regressed)
- `VMODEL_GLM53_EXPERT_BATCH_PREFETCH=1`
- `VMODEL_GLM53_TRUNK_PREFETCH_DEPTH=0..2` (measured at 1)
- `VMODEL_GLM53_TRUNK_PREFETCH_WORKERS=1..2` (measured at 1)
- `VMODEL_GLM53_HOT_PROMPT_KV=1` enables the ordinary target's generic exact
  hot-prefix cache. Native MTP has its own exact paired target/draft boundary.
- `VMODEL_EXECUTION_PROFILE=layers|ops` enables bounded request attribution.

Defaults preserve the original conservative GLM-5.3 profile: expert fetch
batch 1, both prefetch paths off, native MTP off, and generic hot prompt KV off.

## Scope still to prove

- The unmodified captured request renders to 46,849 input tokens and all 134
  real tools. Its one-token capacity gate now completes, but takes 130.76
  minutes of prefill and fails pressure; a full tool/Plex workflow is still
  outstanding and cannot be inferred from max-1.
- Native MTP now has an explicit 65,536-token ceiling and a state-only draft
  prompt prefill that computes exactly the compressed latent retained by
  draft decode, skipping dead prompt Q/O/MLP work. The 8K/64-token real gate is
  the first long admission test; larger context-ladder rungs remain pending.
- The 32K -> 128K -> 256K -> 512K -> 1M context ladder remains outstanding.
- No new GLM-5.3 behavior becomes automatic until token/hash/state gates pass a
  heterogeneous corpus spanning tool counts, system/developer shapes,
  streaming modes, sampling modes, and output lengths.
