# Qwen3.6-35B-A3B: next optimizations after the sub-30 cold gate

Updated 2026-07-26. This is the execution queue after the real 178,616-byte,
134-tool captured request reached **24.9163 seconds cold** and the host-routed
pagination continuation reached **0.2533 seconds warm**.

## Current measured envelope

| quantity | accepted baseline | current automatic path |
|---|---:|---:|
| HTTP wall | 80.7448 s | 24.9163 s |
| input / output | 5,670 / 32 tokens | 741 / 20 tokens |
| prefill | 53.3547 s | 17.4577 s |
| decode | 22.5436 s | 2.2214 s |
| physical weight reads | 29.524 GB | 15.503 GB |
| true Metal peak | 8.618 GB | 7.462 GB |
| useful first call | Plex media list, limit 500 / offset 0 | Plex media list, limit 100 / offset 0 |
| full Plex rubric | 33/100 | 33/100 |

The automatic path is deliberately narrow: fast Qwen MoE, one host-routed
read-only tool, explicit external-action intent, at least 4K characters of
system scaffolding, and no developer message. It projects private execution
onto task history, strips parameter prose, applies request-local top-2 routing,
uses grammar jump-forward, and isolates approximate state in
`gateway_execution_task_top2`.

The project NVMe measures about 1.62 GB/s for uncached sequential reads. Merely
streaming the current 15.503 GB therefore has a best-case serial floor near
9.6 seconds; real routing, non-sequential access, Metal compute, and HTTP/model
startup make that an optimistic floor rather than an attainable prediction.
Substantially beating 20 seconds requires overlap, fewer bytes, or bypassing
model work—not another small cache adjustment.

## Ranked queue

### P0 — activation-aware expert prefetch and cache admission

**Bottleneck:** the automatic run still reads 14.422 GB during prefill. Router
compute, archive reads, decompression/materialization, and expert GEMMs are not
fully overlapped.

Build on the existing expert transition trace rather than adding a second
predictor format:

1. Record per request, layer, phase, routed expert set, router mass/margin,
   source tier, bytes, read start/end, materialization start/end, and eventual
   hit/use.
2. Learn a request-local transition table from prior chunks/tokens. Prefetch
   only the next layer's high-confidence experts, with the governor retaining
   cancellation and admission authority.
3. Rank cache admission by predicted reuse benefit divided by resident bytes,
   not plain recency. Incorrect predictions must be evicted before proven hot
   pages.
4. Keep one physical read stream per device; do not turn prediction into
   same-NVMe queue contention.

This follows MoE-Infinity's sequence-level activation tracing and
activation-aware prefetch/cache design:
[MoE-Infinity](https://arxiv.org/abs/2401.14361).

**Expected range:** 2-5 seconds from the present cold prefill if at least half
of expert read latency overlaps useful router/Metal work. The hard ceiling is
smaller than the 14.422 GB / 1.62 GB/s I/O time because some compute already
overlaps reads.

**Gate:** exact same 741-token prompt and call; five cold ABBA trials; median
wall at least 10% lower; p95 no worse; predicted bytes that are never used
below 10%; peak Metal <=8.5 GB; no swap-out growth; no increase in total bytes
read.

**Stop:** prediction accuracy below 80%, less than 5% wall improvement, or
read-queue contention erases overlap in two independent ABBA sets.

### P0 — quality-aware adaptive expert budget

**Bottleneck:** static top-2 was enough for the captured call, but it is a
blunt global decision inside the private phase and the Plex argument score is
still only 33/100.

Retain request-local routing, but choose the budget per token/layer from router
evidence:

- top-1 only when normalized mass and the top-1/top-2 margin clear calibrated
  thresholds;
- top-2 for the measured default region;
- top-4 or released top-8 for ambiguous routes or named quality-critical
  spans;
- optionally prefer an already-resident near-tie expert only inside a
  calibrated score window.

This combines SQ05's active budget with SQ14's cache-conditional near-tie
routing. SEER-MoE supports the general conclusion that reducing activated
experts needs calibration/fine-tuning to recover quality:
[SEER-MoE](https://arxiv.org/abs/2404.05089).

**Expected range:** another 1-3 seconds and 1-3 GB fewer reads on easy tool
calls; quality recovery on harder arguments may intentionally spend more than
static top-2.

**Gate:** build a replay set spanning read-only shell/filesystem/web/Plex tools,
ordinary no-tool answers, and ambiguous tools. Require selected-tool recall
>=99%, valid arguments >=99%, no more than two percentage points task-score
loss against released top-8, peak <=8.5 GB, and median latency below static
top-2.

**Stop:** top-1 regions do not save at least 5% physical bytes, or recovering
quality causes the adaptive median to exceed static top-2.

### P0 — task capsule with policy retention and argument-quality recovery

**Bottleneck:** removing 15,456 irrelevant system characters produced the
largest speed win, but the model still omitted the requested Plex rating and
root-folder filters.

Replace all-or-nothing system removal with a deterministic capsule compiler:

1. Always retain developer messages, security/confirmation policy, and
   instructions that name the selected tool or its namespace.
2. Drop exact duplicate project instructions and instructions for unavailable
   tools.
3. Keep the latest user request verbatim and keep selected enum/default
   semantics in a short tool contract.
4. Cache the compiled capsule by system hash, selected-tool schema hash, and
   compiler version.

Prompt-compression work shows that relevance-aware context reduction can
reduce latency substantially, but compression overhead and quality must be
measured:
[LongLLMLingua](https://arxiv.org/abs/2310.06839).

**Expected range:** keep cold execution between 25-32 seconds while raising
tool-argument quality; this is primarily a correctness improvement, not a new
latency claim.

**Gate:** preserve the current safety fallbacks; reach at least the captured
planner's required `mediaType`, rating thresholds/operator, excluded root, and
pagination arguments; no latency regression above 20%; no mutating route may
use the task capsule without explicit confirmation policy present.

**Stop:** a generic compiler cannot retain the Plex semantics without growing
above 1,500 input tokens. In that case fix the plugin schema/policy adapter
instead of teaching the runtime domain-specific natural-language parsing.

### P1 — fuse the large MoE operations, not the tiny DeltaNet operations

**Bottleneck:** the previous zmlx conv and gated-norm kernels were 1.38-1.81x
faster in isolation but slowed the real 9B request by 8-10%. Kernel dispatch
overhead dominated because those operations were a small part of a full layer.

Only prototype fusions that remove a materialized intermediate or dispatch
from a measured top-three operation:

- router softmax + top-k + normalized route weights;
- route grouping/scatter metadata;
- MXFP4 expert dequantization + SwiGLU GEMM;
- expert contribution scale + scatter-add.

MLX supports custom Metal kernels through its Python and C++ APIs:
[MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).

**Expected range:** unknown until `VMODEL_EXECUTION_PROFILE=ops` shows at least
15% of engine time in a fusible sequence. Do not inherit isolated zmlx
microbenchmark claims.

**Gate:** start with a one-layer real-weight oracle; compare logits and greedy
tokens; then five full captured ABBA trials. Require >=8% engine-wall gain,
no extra Metal peak, and either byte-identical output or an explicitly named
lossy profile with the same replay quality bar as adaptive top-k.

**Stop:** target sequence is below 15% of wall, compile/dispatch overhead
remains visible in the full trace, or the kernel needs a custom MLX fork.

### P1 — trace-weighted archive layout and vectored expert reads

**Bottleneck:** top-2 still touches thousands of expert pages, and scattered
reads cannot reach the NVMe's 1.62 GB/s sequential result.

Replay the captured expert traces offline and compare:

- current heat-ordered vpack2;
- per-layer coactivation clustering;
- larger authenticated extents containing commonly co-read top-2 pairs;
- vectored/coalesced reads within one shard.

Reuse the transactional F31 reorder/verify/atomic-flip path. Never rewrite the
only authoritative archive in place.

**Expected range:** 1-3 seconds if coalescing materially raises effective read
bandwidth; zero if compute or page reuse already hides the scattered-read gap.

**Gate:** trace simulator first, then authenticated real reads; >=15% fewer
syscalls or >=15% higher effective bandwidth; full hash verification; same
logical tensors and tokens; full-request wall improvement >=5%.

**Stop:** physical bytes increase more than 5%, layout helps one capture but
regresses the held-out trace, or full-request improvement is below 5%.

### P1 — controlled mmap / llama.cpp / Ollama bakeoff

**Bottleneck/question:** explicit `WeightCache` bookkeeping may cost more than
OS page-cache decisions for sparse expert access, but existing external claims
use different quantization and therefore do not answer that question.

Compare vOOM, llama.cpp, and Ollama only after matching:

- source checkpoint and quantization quality as closely as formats allow;
- prompt bytes, chat template, tool schema, output ceiling, and greedy policy;
- cold-file-cache versus warm-file-cache state;
- useful tool call and replay quality;
- physical reads, RSS, swap, CPU, Metal peak, TTFT, and total wall.

Ollama is a packaging/server layer over llama.cpp and is useful only if the
controlled result beats direct llama.cpp or provides operational value.

**Expected range:** unknown. A backend is promoted only from measured
end-to-end wins, never advertised tokens/second from a different quantization.

**Gate:** same useful call, quality within two points, <=8.5 GB Metal, no swap
growth, and >=15% median cold-wall improvement across three ABBA pairs.

**Stop:** quantization/template mismatch prevents a fair comparison, or direct
llama.cpp is not faster; do not spend time wrapping a losing engine in Ollama.

### P2 — resident 9B planner / 35B escalation

Use the fast Qwen3.5-9B task capsule as a semantic planner for tool name and
arguments, escalating to 35B only when confidence/schema validation fails.
This is model substitution, so telemetry and user-facing model identity must
say so; it is not a faster 35B inference claim.

**Expected range:** potentially sub-10-second simple tool calls once resident,
with 35B latency paid only on hard requests.

**Gate:** >=99% selected-tool recall and schema-valid arguments over the replay
set; explicit escalation reason; no hidden reuse of 9B state by the 35B
namespace; cold and resident timings reported separately.

**Stop:** the planner's quality stays below the 35B task capsule, or keeping
both engines resident forces paging/compression on this 16 GB machine.

## Explicitly parked

- **Cache budgets above 5.0 GB:** 5.5 GB crossed the paging cliff and failed;
  7.0 GB reached 9.999 GB Metal. Do not retry without a materially smaller
  transient/KV footprint and a fresh admission proof.
- **Bounded grammar jump batches:** cap-8 raised the full captured wall to
  66.94 seconds and still peaked at 8.94 GB. The current unbounded jump is the
  measured winner.
- **zmlx DeltaNet conv/norm fusion:** keep off; the full request regressed
  8-10% despite attractive isolated microbenchmarks.
- **Top-1 everywhere or expert-pool deletion:** static top-2 has only one
  captured quality proof. More pruning requires the adaptive replay gate.
- **Uncontrolled Ollama comparison:** no conclusion is valid across different
  quantizations, prompts, or tool protocols.

## Recommended execution order

1. Add the P0 activation/I/O trace fields and capture five cold controls.
2. Prototype activation-aware prefetch in the trace simulator, then one real
   request.
3. Build the replay corpus and calibrate adaptive top-k plus task-capsule
   policy retention together.
4. Run an operation-level profile; attempt a large Metal fusion only if the
   measured share clears 15%.
5. Use the same traces for archive-layout simulation.
6. Run the backend bakeoff and 9B cascade only after the native path's P0 work.
