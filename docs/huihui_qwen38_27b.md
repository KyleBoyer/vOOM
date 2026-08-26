# Huihui Qwen3.8 27B on the 16 GB M4

## Installed artifacts

The released checkpoint is pinned to Hugging Face revision
`d42ca8978c5a66e92c3446d46e8adfe03ef692ff`.

- Lossless BF16 source: `models/Huihui-Qwen3.8-27B-abliterated`
- Fast all-MXFP4 derivative: `models/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4`
- Released-BF16 MTP sidecar for the fast target:
  `models/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4/mtp-bf16.safetensors`
- Experimental Qwen3.5-0.8B affine4/group64 proposal sidecar:
  `models/Qwen3.5-0.8B-mlx-all-draft-affine4`
- Quality-oriented mixed derivative:
  `models/Huihui-Qwen3.8-27B-abliterated-mlx-mixed-a8-last4-mtpbf16`
- Internal raw fast tier: `~/vmodel_fast_tier/Huihui-Qwen3.8-27B-abliterated`
- Internal quantized fast tier: `~/vmodel_fast_tier/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4`

`hf cache verify --fail-on-missing-files` passes at the pinned revision. The
source has 1,199 BF16 tensors in 18 shards. The streaming MXFP4 artifact also
has 18 complete shards. Both model directories have an exact row-paged
`embed_rows.bin` sidecar so the 2.54 GB embedding table is not needlessly held
as a monolithic Metal allocation.

The retired `models/Qwen3.6-27B` and
`models/Qwen3.6-27B-mlx-all-mxfp4` directories were permanently deleted. The
unrelated Qwythos checkpoints were not touched.

## Start the recommended fast agent route

```bash
cd '/Volumes/Workspace NVME/git/vOOM'
.venv/bin/python -m runtime.memory_preflight \
  --result logs/huihui-qwen38-fast.preflight.json --sample-seconds 30
.venv/bin/python -u -m runtime.server \
  --profile huihui-qwen38-27b-fast-agent --port 8077
```

Address the fast derivative as:

```json
{
  "model": "lossy-Huihui-Qwen3.8-27B-abliterated",
  "reasoning_effort": "none"
}
```

This is the practical route for the captured 134-tool harness shape. It uses
host-side catalog routing, the full selected tool schema, streaming MXFP4
weights, a 2.2 GB governed weight cache, parallel internal/external tier
fetches, row-paged embeddings, layer-stationary prefill, a 16-layer full-depth
anchor followed by a 1,024-token mixed-depth suffix, chunked DeltaNet, and
target-verified native MTP. The MTP module is the released BF16 tensor set,
loaded only for one speculative round and released before the target sweep.
The named profile uses a top-1 proposal distribution at depth four, masks
greedy proposals with the active grammar, and batches only the
position-independent verifier MLP. For prompts of at least 4,096 tokens it
also primes the released MTP attention cache with 128 target-committed rows
and releases the target head between proposal and verification phases.
Shorter prompts keep the established empty-history/pinned-head path. The
compact MXFP4 language-model head is
pinned and two-layer prefetch is enabled; a trunk pin is not. MTP measures its
draft/target break-even window, temporarily
disengages after a complete unhelpful window, and periodically re-probes so a
poor opening region cannot hide a later predictable span.
Grammar jump-forward and experimental fused DeltaNet kernels stay off because
they do not have an accepted quality case here.

The profile's long-context-only head lifecycle is controlled by
`VMODEL_QWEN35_SERIAL_VERIFY_SUSPEND_LM_HEAD=1`. It remains default-off
outside the named profile. Startup defers the 675.4MB MXFP4
head until first projection; native-MTP copies only its evaluated vocabulary
row through host float32, then the target drops the physical head during each
streamed trunk sweep and zero-copy re-pins it for projection.  On the 16K/64
gate this reduced wall 287.2454s -> 271.0189s and reads 314.125GB -> 275.822GB
with the identical output SHA and a passing pressure gate.  Forcing it on the
6,339-token capture alone regressed 106.1934s -> 107.2490s. Composed with 128
committed MTP rows, however, the capture improved to 98.3885s and a paired
16K/64 run improved 320.5869s -> 297.6953s. The shared 4,096-token boundary
also excludes the measured 1,371-token regression.

A final cold replay through the named production profile (without the
synchronization-heavy `ops` profiler) completed in 98.7772s: 61.0816s
prefill, 34.6881s decode, four target sweeps, 11/16 accepted proposals,
72.297GB of store-accounted reads, 2.916GB peak Metal, and 7.569MB swap-out
growth. It retained output SHA-256 `9b170095...b87b81`. The `ops` profiler is
diagnostic only and must not be used for production wall-time claims.

The captured request is 178,616 bytes with 134 tools, 3 input items, 15,630
message characters, and 162,441 raw tool-schema characters. The fast gateway
turns that incoming shape into a 4,924-token host catalog decision and a
6,339-token selected-tool execution prompt. The optimization sequence retained
the same response SHA-256
`9b170095a170f34a248af480bab274aef28f42ebf1504196bc28cc57e5b87b81`.
With the final depth-1/top-1 policy, a cold streamed 16-output-token replay was
**58.669s prefill, 69.069s decode, and 130.632s wall**. An identical repeat
reused 6,332 prompt tokens and measured **8.838s prefill, 60.795s decode, and
70.287s wall**. Target sweeps fell from the historical 11 to 8, all 8 drafts
were accepted in that replay, and peak Metal was 3.105GB cold / 1.716GB warm.

A bounded exact-scheduling ladder then measured:

| configuration | prefill | decode | wall | result |
|---|---:|---:|---:|---|
| no head pin, no prefetch | 58.669s | 69.069s | 130.632s | control |
| prefetch depth 2 | 58.811s | 67.023s | 128.640s | same hash |
| MXFP4 head pin + prefetch 2 | 60.362s | 62.553s | **125.717s** | same hash; profile setting |
| previous row + 1GB trunk request | — | — | 92.215s to refusal | rejected; unsafe verifier reservation |

The profile now includes the measured depth-four, grammar-aware, batched-MLP
decode stack. To isolate an older depth-two comparison manually:

```bash
VMODEL_QWEN_MTP_GRAMMAR_AWARE_DRAFT=1 \
VMODEL_QWEN_MTP_DEPTH=2 \
VMODEL_QWEN35_MIXED_DEPTH_ENDPOINT_PERSIST=1 \
.venv/bin/python -u -m runtime.server \
  --profile huihui-qwen38-27b-fast-agent --port 8077
```

The first masks each greedy native-MTP proposal with the request's current
grammar before the ordinary target verifier runs.  On the tracked
developer-message/two-tool/max-32 shape this preserved the response hash,
raised native-MTP acceptance from 6/13 to 9/10, and reduced decode from
101.719s to 78.840s.  It remains off in the generic server defaults; the named
Huihui fast profile now opts in after the later multi-shape gates.  Stochastic
sampling retains the established unmodified path.
For depth two, the drafter now advances an independent fork of the grammar
through provisional tokens, so the second proposal is conditioned without
mutating the authoritative verifier state.  Against that same declared
developer/two-tool/max-32 shape, a fresh-process endpoint replay accepted
12/16 drafts over eight target sweeps, preserved the response hash, and
completed in **69.983s wall** (0.006s prefill, 66.407s decode, 2.486GB peak
Metal).  That is 10.3% faster than the 77.977s depth-one endpoint replay.  The
cold depth-two seed took 102.901s because changing MTP depth correctly changed
the cache fingerprint and forced a 34.874s prefill; its decode was 64.619s.
Depth two is retained only as an explicit comparison; the named profile now
uses the later depth-four result.

The same exact recurrent verifier is now bounded at depth four.  It retains a
KDA/KV/hidden endpoint for every possible accepted prefix and commits only the
target-verified prefix; unit oracles cover accepted lengths 0 through 4.  A
fresh-process depth-four replay restored the exact 1,481-token endpoint,
accepted 14/24 drafts, reduced target sweeps from eight to six, preserved the
same response hash, and measured **57.411s wall** (0.006s prefill, 53.802s
decode, 3.557GB peak Metal, 4.260MB swap-out growth).  An in-process repeat was
51.526s.  The result is 26.4% faster than the 69.983s depth-two restart and
56.1% faster than the 130.632s original cold depth-one baseline.  Depth four
remains opt-in globally and is enabled only by the named Huihui fast profile.
A measured depth-five probe
was rejected: its fifth-step acceptance was 0/2, target sweeps rose to seven,
and the cached wall regressed to 63.201s.

The selected profile also passes sustained generation rather than only the
short tool fixtures.  On the deterministic 16,029-input / 64-output gate it
recovered both deep canaries, preserved the historical output SHA, and emitted
nine validation integers.  Relative to the prior depth-one target-verified
gate, depth four reduced target sweeps 37 -> 27 and wall 387.700s -> 335.281s
(13.52%), with 3.857GB peak Metal and 15.286MB swap-out growth.

An additional verifier schedule is available with
`VMODEL_QWEN35_SERIAL_VERIFY_BATCHED_MLP=1`.  It preserves canonical
position order for every full-attention and DeltaNet operation, then evaluates
only the position-independent dense SwiGLU residual for all verifier rows in
one MXFP4 call.  A real MXFP4 operator gate is byte-identical for batched and
row-serial evaluation.  On the depth-four developer/two-tool endpoint it kept
the same output hash, 14/24 acceptance pattern, and six target sweeps while
reducing a clean fresh-process replay from 57.411s wall / 53.802s decode to
**54.417s wall / 50.781s decode** (5.2% / 5.6%).  A second cold shape with 37
input tokens, no tools, no developer message, and a different direct science
prompt emitted the same complete 29-token answer under both schedules; wall
fell from 110.304s to **106.903s** and decode from 100.599s to **97.191s**.
The latter exercised 4,160 verifier positions across 832 batched layer calls.
The setting remains opt-in globally and is selected by the named Huihui fast
profile: the first fingerprint-changing cold 1,481-token
seed completed prefill but hit the unchanged Metal admission ceiling before
decode, whereas its clean endpoint restart passed.  It is a measured schedule
win, not a broadly promoted default.

A depth-four width-two comb-tree follow-up was also rejected.  It rescued one
sibling branch but did not reduce the six target sweeps; the in-process wall
regressed from 51.526s to 55.983s despite identical output.  The serial
depth-four chain therefore remains the selected speculative topology.

Two further composition probes were also rejected.  A depth-four n-gram-first
cascade preserved the output but found prompt/history matches in only 3/8
rounds, accepted 2/12 n-gram proposals, increased target sweeps from six to
eight, and regressed wall from 54.417s to 69.668s while target reads rose from
83.548GB to 107.779GB.  Increasing prefetch depth from the selected two to
three or four likewise did not reduce measured prefetch wait or parallel-tier
service time.  Depth four then failed closed on an immediate repeat when
available memory fell below the configured floor.  N-gram-first remains off
for this workload and prefetch depth two remains the safe measured setting.

Batching the five verifier rows on full-attention layers was also rejected.
It preserved the complete unrelated science-answer hash and the 15/52 native
MTP acceptance pattern, but decode was 98.232s versus 97.191s for the existing
batched-MLP schedule.  Its 208 batched attention calls consumed only 1.391s,
too little leverage to overcome storage/prefetch variance.  No full-attention
batching flag remains in the runtime.

Batching DeltaNet projections/convolution while retaining its per-token FP32
recurrence was rejected as well.  A real MXFP4 oracle bounded activation,
state, and convolution-history drift at every selectable prefix, and the live
science answer and 15/52 MTP acceptance pattern stayed identical.  It was not
a speed win: 624 batched DeltaNet calls consumed 8.551s and decode regressed
from 97.191s to 104.716s.  The experimental runtime flag was removed.

The existing hand-written fused single-position DeltaNet kernel was then
audited against this verifier.  That audit found and fixed a real plumbing
gap: both serial-verifier Qwen branches had omitted the otherwise explicit
`native_fused_deltanet_decode`/zmlx policy, so an initial 113.918s run had not
actually exercised the kernel.  With the policy correctly forwarded, the
complete science-answer hash and the full 15/52 acceptance trace remained
identical, but decode regressed from 97.191s to 104.766s, wall reached
114.907s, and the candidate failed the swap-out gate.  The kernel remains
explicit/off for this profile.  The forwarding fix is retained so the opt-in
means what it says; it is a no-op while disabled.  Evidence:
`logs/qwen38_mtp_depth4_serial_native_fused_science32_20260825.json`.

Compact Delta-factor rollback was rejected for the serial MTP chain.  The
Qwen scalar-gate replay now has an array-equal oracle at every prefix, and the
live candidate retained only 41.073MB of factors while reproducing the exact
science-answer hash, 15/52 acceptance, and 13 target sweeps.  It nevertheless
raised decode from 97.191s to 104.354s, spent 0.740s reconstructing 12
prefixes, and moved peak Metal from 3.589GB to 3.640GB.  Factor capture's
evaluation/storage interference outweighed the smaller rollback payload, so
the serving flag was removed.  The low-level scalar-gate replay correction and
oracle remain for future topology-aware tree work.

A width-two sibling tree at every node of the depth-four native-MTP chain was
also rejected.  It reproduced the exact science-answer hash and accepted
16/104 proposals, one more token than the serial chain, but widened each target
sweep from five to nine positions.  Thirteen sweeps therefore evaluated 117
target positions instead of 65; decode rose from 97.191s to 116.164s, wall to
131.904s, and swap-outs grew 17.203MB.  Batched MLP consumed 22.554s across the
expanded verifier.  The multi-depth tree code was removed; the measured
depth-one sibling tree and the serial depth-four chain remain separate,
explicit candidates.

Ranks-only telemetry on that same unrelated science prompt found three
rank-two root winners in 13 rounds, two later rescuable rejections, and no
useful rank-four candidate.  A narrower follow-up therefore branched only at
the root and retained the ordinary primary chain: six verifier positions, not
nine.  It preserved the exact output, rescued rank two twice, reduced target
sweeps from 13 to 12, and cut total weight reads by 14.396GB.  It still missed
the latency gate: decode was 98.744s versus 97.191s and wall was 111.706s versus
106.903s.  The 72 verified target positions plus 0.817s factor commit erased
the I/O saving, so this candidate was also removed.  Proposal-rank histograms
remain available for future proposal-quality work without retaining prompt,
token, or logit contents.

The recovered Huihui residual direction was then applied directly to the
released-BF16 native MTP attention and MLP residual writers at the measured
1.3 strength.  This genuinely abliterated only the proposal sidecar; the
unchanged target still verified every decision.  It did not improve the draft:
the complete per-depth rank histograms, 15/52 acceptance, 13 target sweeps, and
output hash were identical to the unprojected control.  Decode rose to 106.312s
and wall to 118.617s.  The native-MTP ablation code was removed; the MTP block
does not benefit from inheriting the target's refusal-direction projection.

The second adds a separately typed, checksummed prompt endpoint to the
mixed-depth journal.  It records complete full-attention KV, every DeltaNet
state and convolution history, the prompt logits, and the final hidden row
needed by native MTP.  An identical prompt may restore all of that state; a
strict extension may reuse only the state, while rewinds and branches fail
closed.  A fresh-process replay restored 1,481/1,481 tokens, performed zero
prefill weight reads, emitted the identical function call, and completed in
**77.977s wall** (0.009s prefill, 74.187s decode, 2.452GB peak Metal).  Changing
only the user text correctly rejected the exact endpoint, reused the separate
1,410-token pre-user boundary, and produced a different valid tool call in
102.141s.  Both live rows used the real pinned 134-tool capture as the source;
the declared developer/two-tool scenario and max-32 cap are harness mutations.

The output cap is a benchmark modification and is stated with every result.
The messages and 134-tool incoming catalog retain their captured shape, but
the gateway and mixed-depth suffix are intentionally lossy execution policies.

## Start the released-BF16 route

```bash
cd '/Volumes/Workspace NVME/git/vOOM'
.venv/bin/python -m runtime.memory_preflight \
  --result logs/huihui-qwen38-lossless.preflight.json --sample-seconds 30
.venv/bin/python -u -m runtime.server \
  --profile huihui-qwen38-27b-lossless --port 8077
```

For repeated long requests with a shared exact prefix, use the separately
explicit durable-paging candidate:

```bash
.venv/bin/python -u -m runtime.server \
  --profile huihui-qwen38-27b-lossless-paged-persist --port 8077
```

It journals exact BF16 attention pages plus every DeltaNet/conv state under
`.kv_prompts/huihui-qwen38-27b-lossless`. It accepts only checksummed,
fingerprint-matched stable-boundary extensions; identical state-only endpoints,
rewinds, and divergent branches are rejected. The corrected 49K deterministic
cold/restart gate passes byte identity, but the profile remains opt-in because
building the journal adds a second cold scaffold sweep, consumes 6.3GB on disk,
and the cold construction run exceeded the fixture's strict swap-out limit.

Address the released checkpoint as:

```json
{
  "model": "Huihui-Qwen3.8-27B-abliterated",
  "reasoning_effort": "none"
}
```

The lossless profile retains every released BF16 weight and the released
262,144-token context behavior. It uses only arithmetic-preserving schedules:
parallel raw fast-tier reads, row-paged embeddings, fetch-once
layer-stationary prefill, bounded per-position dense-MLP tiles, exact BF16
paged KV on the external volume, governed weight residency, and target-verified
native MTP. Dense tile scheduling is greedy-token byte-identical to the normal
chunk-major path on the real Qwen3.5 checkpoint. Exact KV paging is also
byte-identical; it changes residency and I/O only.

The live-allocation reserve is 2 GB rather than 5 GB. A 5 GB value was tested
and rejected because it restarted a healthy sweep at layer 9 while Metal was
only about 5 GB. Safety comes from the independently bounded 2.2 GB weight
cache, 768 MB resident KV, 128-row activation tiles, and the project's 8.5 GB
Metal ceiling; the 6 GB post-generation target then sheds consumed LRU weight
pages without changing model arithmetic.

Use `reasoning_effort: none` for ordinary direct answers. Qwen's template
otherwise defaults to thinking: a four-token smoke test emitted the beginning
of hidden planning, while the same request with reasoning disabled returned
exactly `READY` (17 prompt tokens, 2 completion tokens, 96.70 seconds prefill,
153.24 seconds total on cold BF16 weights).

The unchanged released schema renders the captured incoming request to 49,255
tokens with all 134 tools and 166,095 effective schema characters. The repaired
direct `PagedKVCache` constructor now attaches the exact DeltaNet/KDA recurrent
companion. A fresh non-persistent, deterministic full-schema run completed all
64 released-BF16 layers without retry in **3,099.716s prefill / 3,102.732s
wall** for an explicit one-output-token cap. It read 61.738GB of weight bytes,
held 766.378MB of exact paged KV, and peaked at **8.394GB Metal**, below the
8.5GB project ceiling. Its output SHA-256 was
`9e2a5973bcdeb21030f6bb78e6a25cc46c2e1edb080b9832d483c2f9253143bb`.

The cold run is an honest partial gate rather than a blanket pass: the fixture
reported `passed: false` because swap-outs grew 62.587MB against its 16MB
limit, although swap *usage* fell 764.805MB and no retry or correctness failure
occurred. The same greedy request after a fresh process restart loaded the
durable 49,255-token checkpoint from `hot_disk`, performed zero weight sweeps,
and produced the **same output hash**. That restart passed every fixture gate:
10.942s cache load, 10.949s first token, 11.206s engine time, **14.791s wall**,
0.927GB peak Metal, zero weight reads, and 0.885MB swap-outs. This is a 209.8x
wall-time improvement for the exact repeated prefix. The journal is 6.3GB / 777
files with no partial files. Use the ordinary lossless profile for cold or
one-off traffic and the explicit persistent profile for genuinely repeated
long prefixes.

The earlier `paged_v4` artifact remains historical only: it omitted the KDA
companion and reset recurrent state at tile boundaries. Its timing must not be
used as a lossless correctness result.

## Lossy-answer acceptance rule

Quantization and mixed-depth prefill are admitted only behind the named fast
profile. A focused two-page Plex quality run proved that the model planned the
right tool twice with the right filters and offsets. Its own terminal summary
nevertheless scored only 87.5/100: it leaked two rejected titles and was
truncated. That answer is rejected.

Running those same model-produced calls through
`evaluate_plex_policy_adapter` produces the exact eligible set
`ALPHA_G`, `CHARLIE_TVY`, `BRAVO_PG13`, and `DELTA_TVY7`, excludes every
ineligible title, and scores 100/100. This specialist-plan plus deterministic
policy-adapter boundary is the accepted fast path for filtered, paginated
answers. A separate generic model-only terminal-synthesis experiment returned
no final-channel text and is also rejected.

The rule is therefore simple: free-form fast answers are allowed only after
their task-specific quality gate passes. For externally verifiable filtering,
pagination, totals, or policy enforcement, keep execution and final selection
deterministic; do not promote the model's unchecked prose.

## Optimization results and stop decisions

The `100/100` score is not a score for the lossless BF16 checkpoint. It is the
whole-visible-output score of deterministic rendering over verified model tool
calls. Direct all-MXFP4 terminal prose scored 87.5/100. A fresh live run with
the current profile made the two correct Plex calls, fetched offsets 0 and
200, and then rendered the 41-token final answer in 0.0077s with zero model
prompt, prefill, or decode. The whole-visible scorer found all four eligible
titles and no forbidden title: 100/100.

Implemented and measured items:

- **Released-BF16 MTP sidecar:** 15 tensors, 849,398,784 payload bytes, byte-
  equal to the pinned source. Round-local lifetime prevents the sidecar from
  colliding with the target verifier. The initial depth-1 `flat-k1` gate
  reduced the historical 11 target sweeps to 8. The later exact-state depth-4
  verifier, grammar-aware proposals, and batched verifier MLP passed the
  unmodified capture plus developer/two-tool and no-tool science shapes and
  are now composed only in the named fast profile.
- **Exact compiled DeltaNet segments:** array-equal output/state gates pass at
  lengths 1/2/31/32/33/64/127/128 and split boundaries. A Huihui-shape
  weights-free probe measured 54.708ms sequential versus 42.144ms compiled
  (1.298x). The real released-BF16 cold A/B was byte-identical over four
  greedy output tokens, but prefill was 24.122s versus 24.107s and wall was
  102.955s versus 102.707s. It therefore remains an exact, explicit long-
  context candidate rather than a promoted short-request speed path.
- **Fused stable-boundary/scaffold prefill:** the layer-stationary path can
  process a cacheable chat boundary and its short generation scaffold in one
  exact fetch-once sweep. Oracle tests require byte-identical logits, recurrent
  state, and boundary snapshots against the split path. Durable paged journals
  deliberately disable this fusion because they must commit the scaffold-free
  boundary before proceeding; that is why first-time persistent construction
  costs an additional sweep while ordinary lossless traffic does not.
- **Native serial Metal GDN:** a microbenchmark measured about 9.99x for the
  recurrence, but it changes FP32 association. On the real capture it kept the
  response hash and reduced prefill from 58.669s to 56.742s; total wall was
  127.692s. It remains lossy/opt-in because one matching output is not a corpus
  proof. Composed with head pinning and prefetch depth 2, it again retained the
  hash and reduced prefill to 56.262s, but wall was only 0.335s below the
  125.717s control and the run exceeded the strict swap-out gate by 39,936
  bytes. That composition is not promoted.
- **Durable hybrid prefix checkpoints:** checksummed full-attention K/V plus
  every DeltaNet state and convolution history survive process restart. A real
  Qwen3.5-4B hybrid gate compared byte-identical tokens and measured 2.233x
  restart, 2.007x fork, and 2.653x disk-recovery speedups at 6.930GB peak Metal.
- **Mixed precision:** the built 23.082GB logical artifact keeps attention in
  MXFP8, the final four layers and MTP in BF16, and the remainder in MXFP4. It
  loads and generates correctly, but a matched 34-input/8-output gate was
  73.309s versus 42.270s for all-MXFP4. It is a quality experiment, not the
  speed route.
- **Exact row-paged BF16 head reranking:** K=64 reads only 655,360 exact bytes
  per decision plus sorted/coalesced metadata. An initial `1/1` real probe
  exposed stale MTP accounting rather than a verifier bypass; request-wide
  accounting and position-correct constrained reranking are now fixed. The
  repaired short run measured exact-winner recall@64 of 9/9 on authoritative
  target positions and 4/4 on draft positions, with one target winner changed.
  Privacy-safe ranks-only capture and a fail-closed evaluator now enforce a
  fixed K=64 / at least 1,000 authoritative target positions / 100% recall /
  at least 8 requests and 6 heterogeneous shapes gate. Draft projections and
  constrained provisional projections are excluded. No qualifying real corpus
  has been captured yet, so the path remains explicitly unpromoted.
- **Multi-request layer-stationary scheduler:** the standalone dense/MoE API
  fetches each trunk layer once for independent requests and shares physical
  LM-head row reads. The explicit default-off
  `POST /v1/qwen/layer-stationary/completions` route gives every request private
  KV/KDA state and currently admits deterministic greedy requests only. A real
  bounded Qwen3.5-4B gate returned two heterogeneous outputs identical to
  serial execution in 9.811s versus 10.861s, with 64 layer gets instead of the
  serial-equivalent 96. It is an aggregate-throughput path, not a single-
  request latency claim.

Measured STOP decisions:

- Page-native online attention saved about 203MB and was 2.555x faster for a
  49,255-key single-position decode, but every shape failed BF16 byte identity
  and the 128-position prefill tile was 4.08x slower. No lossless runtime path
  was enabled.
- The bounded Metal-I/O trace found CPU-to-MLX staging was only 3.4–5.4% of
  the device pipelines; concurrent raw-plus-stage throughput was within noise
  of raw reads. It cannot clear the 10% implementation gate.
- A 1GB trunk pin planned two layers / 0.383GB but caused a safe verifier
  reservation refusal. The head-only pin is retained; trunk pinning is off.

## Large-context and sustained-output validation (2026-08-24)

The old 16-token captures were scheduling probes, not realistic generation
quality gates. A new deterministic HTTP fixture places two canaries at 13% and
73% of a synthetic long prompt, requires their exact recovery, and records
input/output counts, hashes, timings, KV geometry, Metal peak, available
memory, and swap-out growth without retaining the private prompt or answer.

The work exposed and fixed four generic long-context issues: mixed-depth KV
admission now uses the real retained layer geometry and FP32 runtime KV dtype;
the explicit Qwen prefill ceiling applies even with durable persistence;
dense layer-stationary prefill releases/eagerly consumes evaluated tile
temporaries; and transient history is keyed by the full layer-stationary
sequence shape rather than tile width alone. Exact real-checkpoint oracles
remained byte-identical after the lifetime/scheduling change.

The fully passing sustained gate used `VMODEL_QWEN35_HOT_KV=0` to avoid
retaining a second reusable boundary during a one-shot long generation. This
strict, identity-bearing opt-out defaults to `1`, so ordinary serving behavior
is unchanged. Results: 16,029 rendered input tokens, exactly 64 output tokens,
both canaries, exact required prefix, 9 consecutive validation integers,
144.034s prefill, 424.717s decode, 568.751s engine / 569.732s HTTP wall,
3.580GB peak Metal, 36.323MB swap growth, and 6.557GB available at completion.
The output hash matched the earlier cached-state semantic run exactly.

That run also exposed a decode-policy bottleneck: MTP rejected the first three
tokens and then stayed disabled for all 60 remaining sweeps.  The measured
adaptive policy now waits for the break-even window (14 rounds in the replay)
and periodically re-probes.  The same fresh-process 16,029/64 request accepted
26/37 verified proposals and required only 37 target sweeps.  Decode fell to
260.446s, engine time to 386.718s, and HTTP wall to 387.700s; output SHA-256 was
unchanged.  Before each known 849.399MB sidecar allocation, consumed target
pages are evicted using the exact header size.  That made the final gate
pressure-clean at 4.015GB peak Metal, 16.564MB swap-out growth, and 6.365GB
available, without a decode regression versus the unguarded corrected run.

A separate no-hot 30,029-input / 32-output run recovered both canaries with
the exact prefix and marker at 4.055GB peak Metal and 28.754MB swap growth;
prefill was 348.918s and decode 222.237s. It is a large-context retrieval and
pressure pass, but the composite sustained-format fixture remains red because
the 32-token truncation fit only one of the requested two validation integers.
Do not report that artifact as a full composite pass. The combined 30K/128
attempt also remains a STOP: prefill completed, but decode hit the memory floor
and swap growth reached 706MB. The released model advertises 262,144 context;
this work validates 30K on this machine, not the advertised maximum.

## Depth-4 n-gram/MTP cascade and pressure follow-up (2026-08-26)

`VMODEL_QWEN_MTP_NGRAM_FIRST=1` now composes with the existing native-MTP
depth-four chain. A prompt/history suffix match proposes up to four tokens
without loading the 849.399MB BF16 MTP sidecar; a miss falls back to the
ordinary depth-four drafter. Both sources use the same authoritative target
verifier and recurrent/KV rollback. The setting remains explicit and defaults
off.

On a tracked but intentionally modified non-streaming/no-tool repetitive
scenario (84 rendered input tokens, 10 output tokens, temperature zero), the
cascade preserved output SHA-256
`c886b3783fc717ff9f3e021e40cb1c2699068f51512d601f0641ebb97d8a4d05`.
Two n-gram matches accepted five tokens, cut target sweeps 5 -> 4, BF16 sidecar
round loads 5 -> 2, and total streamed weight bytes 82.094 -> 67.215GB. Wall
fell 50.7372 -> 39.4793s (22.2%). This number depends on the stated prompt and
tool-shape mutation; it is not the unmodified 134-tool result.

The unmodified captured 134-tool request also preserved its known response
hash and found two n-gram matches, but a host-pressure prefill retry selected
32-token chunks and made that cold sample 175.0009s versus the same-day
106.1934s control. Because n-gram lookup begins only after prefill, this is
evidence of cold scheduling variance rather than prompt-dependent prefill
work, but it is still a promotion failure. The named fast profile therefore
does not enable the cascade.

The fixed 16,029-input / 64-output fixture gave the strongest sustained-decode
result when the 675.441MB MXFP4 LM head was left demand-cached instead of
pinned. It reproduced the known output SHA-256
`bdff23c8cc7742c78896798a571c59112be9b39559ca20b16e577b3550b44fca`,
both distant canaries, the exact required prefix, and nine validation integers.
N-gram-first accepted 20/24 proposals across six matches; combined acceptance
was 45/72, target sweeps fell 27 -> 18, sidecar loads fell 27 -> 12, and wall
fell 335.2813 -> 287.2454s (14.3%). It is not promoted: swap-out growth was
19.841MB, above the 16MB gate. A stricter 5.35GB floor still emitted identical
bytes in 307.5602s but used 18.432MB swap; 5.4GB failed closed by 10MB during a
later verifier reservation. These are explicit long-context experiments, not
production defaults.

A proposal-only affine4 -> affine2 requantizer was added for dense Qwen3.5
draft artifacts. The real 0.8B derivative shrank 493.916 -> 276.009MB (44.1%),
but both the affine4 release-before-target path and affine2 retained path were
refused by the unchanged verifier memory ceiling. Affine2 also increased live
allocator pressure despite its smaller file. It is a reproducible storage tool,
not an accepted serving sidecar. Prefetch depth four and a proposed 64-token
prefill retry rung were likewise rejected by live gates; the production
prefetch depth remains two and the conservative 128 -> 32 retry is unchanged.

## Tiled paged decode and selective low-margin continuation (2026-08-26)

Two additional paths are implemented as explicit experiments.  Neither is a
named-profile default.

`VMODEL_QWEN_MTP_SELECTIVE_TREE_MARGIN=2` forks the native MTP recurrence only
at a low-margin non-root proposal, generates one rank-two continuation, and
passes both chains through the target-authoritative verifier.  On the untouched
134-tool/max-16 capture it preserved SHA `9b170095...b87b81`, reduced target
sweeps 4 -> 3 and decode 32.4705s -> 29.7882s in a same-session comparison.
Total wall regressed 99.3253s -> 101.8428s because prefill varied upward.  At
16K it required a 2.03GB tree-verifier transient and failed closed, so it is a
short-context research lever only.

`VMODEL_QWEN35_PAGED_ONLINE_ATTENTION=1` is the larger long-context win.  For
one-token decode it consumes exact BF16 spill pages in bounded tiles and uses a
fused Metal online-softmax kernel; multi-token prefill and the lossless path
still use the established MLX SDPA.  The changed reduction order makes this a
lossy path even when a tested output happens to be byte-identical.  The winning
measured settings are:

```
VMODEL_QWEN35_KV_MAX_MB=64
VMODEL_QWEN35_KV_PAGE_POSITIONS=1024
VMODEL_QWEN35_PAGED_ONLINE_ATTENTION=1
VMODEL_QWEN35_PAGED_ONLINE_TILE_POSITIONS=2048
VMODEL_QWEN_MTP_SELECTIVE_TREE_MARGIN=0
```

On the fixed 16,029-input / 64-output gate these settings reproduced SHA
`bdff23c8...44fca`, both canaries, the exact prefix, and nine validation
integers.  Wall was 260.0515s, decode 128.5402s, peak Metal 2.650GB, and
swap-out growth 13.058MB.  Relative to the earlier successful committed-history
run, wall improved 297.6953s -> 260.0515s (12.6%) and decode 168.0094s ->
128.5402s (23.5%).  Coarsening spill pages from 256 to 1,024 reduced reloads
24,023 -> 5,964 and reload time 17.8079s -> 9.6559s.  A 2,048-page/300MB arm
was rejected despite fewer reloads: it regressed wall to 268.5371s and exceeded
the swap gate.  The untouched 134-tool/max-16 replay preserved its known SHA
but regressed 98.7772s -> 102.5523s, which is why automatic short-context use is
not allowed.

## Evidence

- FreeToken audit, exact MTP two-SSD split, verifier/prefetch instrumentation,
  and rejected CPU/body-rebalance experiments:
  `docs/huihui_qwen38_freetoken_improvements.md`
- Fast captured-shape gate: `logs/huihui_qwen38_fast_capture16_final.json`
- Final cold/warm MTP gate:
  `logs/huihui_qwen38_fast_mtp1_bf16_flat1_capture16_cold_warm.json`
- Live whole-visible quality gate:
  `logs/huihui_qwen38_fast_plex_policy_live.json`
- Head/prefetch winner: `logs/huihui_pin_H1_T0_P2_capture16.json`
- Rejected trunk pin: `logs/huihui_pin_H1_T1000_P2_capture16.json`
- Native GDN gate: `logs/huihui_native_gdn_capture16.json`
- Mixed artifact gate: `logs/huihui_mixed_a8_last4_short_direct8.json`
- Repaired row-head diagnostic:
  `logs/huihui_rerank64_short8_recall_postfix.json`
- Released-BF16 compiled A/B:
  `logs/huihui_qwen38_lossless_compiled_ab_baseline.json` and
  `logs/huihui_qwen38_lossless_compiled_ab_candidate.json`
- Durable hybrid-prefix live gate:
  `logs/huihui_qwen38_qwen4b_hotkv_live.json`
- Corrected released-BF16 49K deterministic cold gate:
  `logs/huihui_qwen38_lossless_capture1_paged_corrected_greedy.json`
- Exact persisted 49K restart gate:
  `logs/huihui_qwen38_lossless_capture1_paged_persist_restart_greedy.json`
- Real bounded multi-request gate:
  `logs/qwen4b_multi_request_live_bounded.json`
- Native-GDN plus head-pin/prefetch composition:
  `logs/huihui_native_gdn_h1p2_capture16.json`
- Focused quality gate: `logs/huihui_qwen38_fast_plex_focused_specific.json`
- Passing 16K/64 sustained gate:
  `logs/qwen38_large_context_16k_out64_nohot.json`
- Passing 16K/64 adaptive-MTP/cache-overlap gate:
  `logs/qwen38_large_context_16k_out64_mtp_recovery_cache_prepare.json`
- N-gram repetitive A/B:
  `logs/qwen38_ngram4_repetition_baseline32_20260826.json` and
  `logs/qwen38_ngram4_repetition_candidate32_20260826.json`
- N-gram unmodified capture correctness/pressure sample:
  `logs/qwen38_ngram4_unmodified_capture16_20260826.json`
- N-gram 16K/64 unpinned-head near-pass:
  `logs/qwen38_ngram4_unpinned_head_large16k_out64_20260826.json`
- Winning tiled-paged 16K/64 gate:
  `logs/qwen38_paged64_page1024_online2048_large16k_out64_20260826.json`
- Rejected 2,048-position page arm:
  `logs/qwen38_paged300_page2048_online2048_large16k_out64_20260826.json`
- Exact untouched captured-shape tiled-paged gate:
  `logs/qwen38_paged64_online2048_exact_greedy_capture16_20260826.json`
- Winning depth-12 plus tiled-paged 16K/64 gate:
  `logs/qwen38_suffix12_anchor128_paged_large16k_out64_20260826.json`
- Untouched captured-shape depth-12 quality sample:
  `logs/qwen38_suffix12_unmodified_capture16_20260826.json`
- Rejected heterogeneous developer/tool depth-12 and depth-14 gates:
  `logs/qwen38_suffix12_developer2tools_noonline_capture16_20260826.json` and
  `logs/qwen38_suffix14_developer2tools_noonline_capture16_20260826.json`
- 30K/32 retrieval result (composite red only for second integer):
  `logs/qwen38_large_context_30k_out32_nohot.json`
- 30K/128 pressure STOP:
  `logs/qwen38_large_context_30k_out128_shapeclass.json`
- Historical invalid-KDA full-schema artifact:
  `logs/huihui_qwen38_lossless_capture1_paged_v4.json`
- Profiles: `profiles/huihui-qwen38-27b-fast-agent.yaml`,
  `profiles/huihui-qwen38-27b-fast-long-context.yaml`,
  `profiles/huihui-qwen38-27b-lossless.yaml`, and
  `profiles/huihui-qwen38-27b-lossless-paged-persist.yaml`
