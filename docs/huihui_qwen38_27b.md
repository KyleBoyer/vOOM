# Huihui Qwen3.8 27B on the 16 GB M4

## Installed artifacts

The released checkpoint is pinned to Hugging Face revision
`d42ca8978c5a66e92c3446d46e8adfe03ef692ff`.

- Lossless BF16 source: `models/Huihui-Qwen3.8-27B-abliterated`
- Fast all-MXFP4 derivative: `models/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4`
- Released-BF16 MTP sidecar for the fast target:
  `models/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4/mtp-bf16.safetensors`
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
Calibration selects a top-1 proposal distribution at depth 1. The compact
MXFP4 language-model head is pinned and two-layer prefetch is enabled; a trunk
pin is not. MTP automatically disengages after repeated draft rejections.
Grammar jump-forward and experimental fused DeltaNet kernels stay off because
they do not have an accepted quality case here.

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
  colliding with the target verifier. Depth-1 `flat-k1` reduced the historical
  11 target sweeps to 8. Depth 2 saved no additional sweep and was 1.8s slower,
  so it remains opt-in and the profile stays at depth 1.
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
- Historical invalid-KDA full-schema artifact:
  `logs/huihui_qwen38_lossless_capture1_paged_v4.json`
- Profiles: `profiles/huihui-qwen38-27b-fast-agent.yaml`,
  `profiles/huihui-qwen38-27b-lossless.yaml`, and
  `profiles/huihui-qwen38-27b-lossless-paged-persist.yaml`
