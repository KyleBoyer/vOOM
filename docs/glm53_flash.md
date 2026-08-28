# GLM-5.3-Flash on the 16 GB M4

Status date: 2026-08-28

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

## Explicit controls

All new narrow speed paths remain opt-in pending the heterogeneous real-request
corpus required by the anti-overfit policy:

- `VMODEL_GLM53_MTP=1`
- `VMODEL_GLM53_MTP_DEPTH=1..5` (measured at 3)
- `VMODEL_GLM53_MTP_MAX_PROMPT_TOKENS=1..2048`
- `VMODEL_GLM53_EXPERT_FETCH_BATCH=1..8` (measured at 8)
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
  real tools. It has not yet completed a GLM-5.3 response/quality score. A cold
  prompt necessarily streams most of the 328 GB checkpoint, so its first-call
  floor is materially above 90 seconds. It must be reported separately from
  exact repeat/state-cache latency.
- Native MTP is deliberately limited to 2,048 prompt tokens until the long
  draft-context/index-sharing oracle passes. Larger prompts fall back to the
  exact target rather than silently using an unproved draft path.
- The 32K -> 128K -> 256K -> 512K -> 1M context ladder and sustained larger
  output-token gates remain outstanding.
- No new GLM-5.3 behavior becomes automatic until token/hash/state gates pass a
  heterogeneous corpus spanning tool counts, system/developer shapes,
  streaming modes, sampling modes, and output lengths.

