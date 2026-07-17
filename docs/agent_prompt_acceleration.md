# Agent-prompt acceleration on small Macs

vOOM has two deliberately separate mechanisms for large tool-using prompts:

- `lossy-<model>` compacts tool schemas and keeps one exact, in-memory prompt KV
  branch. It targets latency within the checkpoint's released context window.
- `lossy-long-<model>` adds a named static-YaRN profile. It targets capacity
  beyond the released window and is not a speed optimization.

Both are side-quest profiles. The ordinary model ID retains the released prompt
representation and weights.

## Compact schemas plus exact hot-prefix reuse

Fast mode keeps all tools by default. It sorts/minifies the JSON representation
and removes nested schema annotations such as descriptions, titles, examples,
and defaults, while preserving tool names, top-level selection descriptions,
property-map keys, types, required fields, enums, const values, unions, bounds,
and `additionalProperties`. Object-valued `enum` and `const` entries are opaque
JSON literals and are never pruned. Rendering uses Jinja's HTML-safe JSON escape
contract, so schema text cannot close a template delimiter.

This transform is lossy because whitespace and annotations are model input. It
has a useful cache property: edits confined to removed annotations produce the
same compact prompt. The original unmodified schemas remain available to the API
response and tool executor.

After a fast request, the engine retains a small LRU of recent KV states
(`RuntimeConfig.hot_prompt_kv_slots`, default 1; set the server-level default
via `VMODEL_HOT_PROMPT_KV_SLOTS=N`) rather than synchronously copying them to
disk. The next request computes the token longest-common prefix against
every retained slot (not just the most recent one) and reuses whichever
gives the best match. Raise this above 1 if a harness interleaves unrelated
prompts (e.g. a title-generation or working-memory call) between turns of
the same conversation — with only 1 slot, a real interleaved request found
live (2026-07-14) evicted the main conversation's retained state before its
next turn could ever reuse it, so that turn also paid a full cold prefill
(26,907 tokens, twice, ~52s each). Size this to the actual number of
concurrently-live prompt lineages a caller's harness produces: each slot
holds a full KV state proportional to its own context length (~1.26 GB for
a real 44K-token Qwen2.5-1.5B conversation), so this is a real memory
tradeoff against the governor's sampled live Metal headroom, not a free win.
With the current 4,096-token chunks, a
representative 131-tool replay measured:

```text
prompt tokens:              28,728 -> 28,751
token LCP:                  28,713
exact reusable watermark:  28,672
second-request suffix:      79 tokens
linear suffix fraction:     0.274773%
logical causal-pair work:   0.548782%
```

Confirmed live against a real harness (2026-07-15, `hot_prompt_kv_slots=2`):
consecutive turns of one real conversation reused 88-91% of the prompt from
memory (24,576 of 26,967-28,055 tokens), each completing in 7-14s total
instead of the ~52-67s a cold prefill of the same size takes.

A later live session the same day showed `hot_prompt_kv_slots` alone is not a
safe fix: the harness sent a VARIABLE number of tiny non-conversational calls
(title generation, working-memory updates -- 89 and 885 tokens, `tools=0`)
between real conversation turns (26,872-27,047 tokens, `tools=131`) -- one
interleaved call between one pair of turns, two between the next. With only
2 slots, the pair of interleaved calls filled both, evicted the main
conversation's own state, and its next turn paid a full cold prefill again
(54s) exactly like the original single-slot bug. Raising the slot count
further only sets a higher number a busier session can still exceed. Fixed
with `RuntimeConfig.hot_prompt_kv_min_tokens` (server default via
`VMODEL_HOT_PROMPT_KV_MIN_TOKENS=N`, default 2,048): a prompt shorter than
this is never RETAINED as a slot (lookup/matching against existing slots is
unaffected), so tiny side calls can never evict the expensive conversation
state no matter how many arrive between turns.

### Surviving a restart, and forking

The LRU above is pure in-memory by default and does not survive a server
restart -- every slot is gone and the next request for any conversation pays
a full cold prefill again. Set `VMODEL_HOT_PROMPT_KV_PERSIST_DIR=<dir>`
(server default: unset/disabled) to back it with disk: engine startup reloads
up to `hot_prompt_kv_slots` conversations before the first request arrives,
so they can resume warm across a restart.

The backing store (`runtime/hot_kv_persist.py`) is a parent-hashed segment DAG.
F37 (`runtime/kv_store.py`) now wraps the same durable substrate; the former
growing full-snapshot implementation remains only as format history. This closes
the double-write/O(n^2)-per-request behavior that originally made fast mode
disable synchronous F37 writes:

- A **segment** is an immutable, content-addressed KV delta. Its canonical ID
  covers the fingerprint, parent, new token IDs, KV layout, payload bytes, and
  payload SHA-256 -- the newly computed keys/values for one span on top of a
  specific parent segment (or none, for a conversation's first chunk).
- A **checkpoint** is a small pointer record (endpoint logits, prompt
  length, the reusable-prefix watermark) with its own immutable payload hash
  and generation ID; multiple endpoint generations may reference one leaf.
- "Append" is simply writing a new segment whose parent is the old leaf --
  no existing file is ever rewritten or truncated, so per-turn write cost is
  the actual new bytes only (true O(delta)), not the whole conversation
  again. Content-addressing also means two turns, or two DIFFERENT
  conversations, that happen to produce the same delta on the same parent
  (e.g. a shared system-prompt/tool-schema prefix) never duplicate that
  segment on disk.
- GLM segments also carry the matching DSA index-key delta. The cache-format
  identity is versioned: older segments that stored compressed MLA without
  DSA keys are ignored, because an exact first logit could hide divergence on
  the second token of a restored extension.
- Payload-first fsync publication plus immutable manifests makes crashes leave
  either a complete generation or an uncommitted orphan. Every selected payload
  is checksum-verified; corrupt newest generations fall back to older valid
  ones. Cross-process locks, reader leases, dead-lease cleanup, and count/byte
  bounded mark/sweep GC cover concurrent readers and retention.
- **Forking falls out for free**: two divergent continuations from the same
  parent produce two sibling child segments, both valid. Consuming a
  checkpoint in memory (a new turn matches and supersedes it) deliberately
  does NOT delete it from disk -- disk checkpoint retention has its own,
  separate, larger budget (`hot_prompt_kv_persist_max_checkpoints`, default
  64, oldest-by-mtime evicted past that), decoupled on purpose from
  `hot_prompt_kv_slots`'s in-memory capacity. Otherwise a later, different
  continuation from an earlier point -- a "regenerate," an edited earlier
  message -- could never find that point again, only whatever branch
  happened to consume it in memory first.
- **Fixed (2026-07-15, later still)**: the "repeat" reuse case (a request
  whose tokens exactly equal a prior request's PROMPT, before that prior
  request's own generation -- e.g. N agentic/cron tasks all starting from
  one identical preamble, each generating independently) used to land at a
  position buried inside one merged tail segment, with no addressable node
  to fork from there. Fixed by splitting that tail into two segments: a
  **prompt-tail** segment ending exactly at the prompt's own length, and a
  separate **generation** segment for the model's own continuation. Now a
  "repeat" match forks its own generation segment directly off the shared
  prompt-tail parent -- the same O(delta), fork-preserving treatment as
  "branch" and "endpoint" already had. Fixing this surfaced a real bug: the
  save path had derived how many tokens a parent chain covers as
  `len(parent_chain) * chunk_size`, which only holds when every segment in
  the chain is a full chunk -- true for "branch," false once "repeat"'s
  parent chain can end in a shorter prompt-tail segment. Now the caller
  passes the exact covered length explicitly (the same `matched` value the
  in-memory lookup already computed) rather than re-deriving it.

Verified with `tests/test_hot_prompt_kv.py`:
`test_hot_prompt_kv_persists_across_engine_restart` (the restart round trip),
`test_checkpoint_retention_is_recency_bounded_not_lru_bounded` (disk budget
is its own policy, not mirrored from the in-memory LRU),
`test_forking_keeps_a_consumed_checkpoint_retrievable` (a consumed
checkpoint survives a fork, and shared ancestor segments are proven
byte-identical/never rewritten via unchanged mtimes), and
`test_repeat_case_forks_independent_generations_off_shared_prompt` (two
independent continuations of one identical prompt fork to two different
leaves, both independently resumable, sharing every byte of the prompt).
`tests/test_hot_kv_durability.py` additionally proves same-size corruption
detection, salted repair generations, older-generation fallback, immutable
inode/content reuse, crash boundaries, live/dead reader leases, byte-budget GC,
delta-only extension writes, and one concatenate per reconstructed tensor.
The real Qwen2.5-1.5B MXFP4 gate in
`tests/fixtures/qwen_hot_kv_live_gate.py` also passed restart, edited-suffix
fork, regenerate, and one-slot eviction recovery with identical token IDs:
3.33x restart, 3.98x fork, and 5.95x disk-recovery speedups at 2.63GB peak.

**Closed (2026-07-15, later still)**: on a total in-memory miss (no slot in
`_hot_prompt_slots` matches at all), `generate()` now calls
`HotPromptKVPersistence.find_best_match()` before falling back to a cold
prefill -- a metadata-only scan (no tensors loaded) that reconstructs each
checkpoint's full token list from its chain's small `.seg.json` files and
scores it with the identical repeat/endpoint/strict-extension/branch logic the in-memory
loop uses. Only the actual winner's tensors get loaded
(`load_matched_chain()`, exactly `n_segments` of its chain, never more).
This deliberately does not compete with an in-memory hit -- it only fills
the gap when memory has nothing at all, e.g. more concurrent agentic/cron
tasks sharing one preamble than fit in `hot_prompt_kv_slots`, where an
earlier task's shared prefix is still sitting on disk. Reported as
`path_stats["prompt_cache_source"] == "hot_disk"` (a new, distinct label
from the in-memory `"memory"` and F37's own `"disk"`) with
`hot_prompt_kv_disk_hit=1`. Fixed a real, if not-yet-reachable-in-production,
interaction bug while wiring this in: the existing F37 disk-fallback check
only excluded itself when the source was `"memory"`, so it would have
happily overwritten a successful `"hot_disk"` match with a worse (or
unrelated) F37 result; both `"memory"` and `"hot_disk"` are now excluded.
(In the actual server config this pairing is inert either way -- fast mode,
the only mode that enables `hot_prompt_kv`, always sets F37's
`prompt_kv_dir=""` -- but the engine API allows both, so this needed
fixing regardless.) Verified with
`tests/test_hot_prompt_kv.py::test_disk_fallback_recovers_a_task_evicted_from_the_in_memory_lru`:
task 1 runs, gets evicted from a `slots=1` in-memory LRU by an unrelated
request, and a later repeat of task 1's own prompt recovers via disk
(`hot_disk`, exact hit) instead of silently recomputing cold. The same real-Qwen
gate evicted one task from the one-slot LRU and recovered it from this disk path
with identical IDs and a 5.95x speedup.

Separately, F37's longest-prefix view of the same journal has its own byte budget,
`RuntimeConfig.prompt_kv_max_mb` (default 2,000 = 2 GB). Set the server-level
default via `VMODEL_PROMPT_KV_MAX_MB=N` (megabytes; 0 = unbounded). Extending a
known prefix writes only delta segments, while an endpoint whose complete
reachable chain exceeds the budget is skipped. On first v6 use, obsolete v5
full-snapshot pairs are swept once so they cannot sit outside the new budget.

These are CPU replay/accounting results, not a claimed latency speedup. A live
greedy A/B must still prove matching output/tool behavior, cache-path telemetry,
TTFT, total wall time, and a Metal peak below the governor ceiling sampled for
that run.

An identical prompt can reuse the complete prompt and its saved endpoint logits,
even after the prior request generated multiple tokens. A related prompt can
branch only at a watermark actually produced by prefill; decode tokens and
arbitrary semantic module boundaries do not create reusable checkpoints.

Set `VMODEL_FAST_TOOL_LIMIT=N` to enable a deterministic soft top-N shortlist.
It is disabled by default. Tools explicitly named by the user or used in prior
assistant tool calls are hard-pinned and may exceed the configured soft limit.

## Exact prefix DAG versus detached capsules

Fast mode now implements the exact version of tool capsules as canonical catalog
versions (`tool_order_profile=canonical-name-v1`). The prompt-only tool copy is
sorted by function name after shortlist selection and schema compaction; protocol
responses and execution retain request order. Shortlist score ties use the same
canonical name order, duplicate function names fail deterministically, and
`tool_catalog_id` exposes the compact catalog fingerprint. Lossless mode retains
request order byte-for-byte.

This turns any permutation of an identical tool set into an identical rendered
prompt/token sequence. Added or removed tools reuse the ordinary prefix only up
to the first changed canonical schema; they do not splice later KV back in. A
131-tool Qwen tokenizer proof measured:

```text
                                      request-order       canonical-name-v1
random permutation LCP                     62 tokens             all 25,950
reusable 4,096-token chunks                 0                    all 6 chunks
middle add/remove reusable prefix           n/a                  12,288 tokens
```

Tool *history* is canonicalized independently of catalog order. Parallel tool
results are rendered in the assistant turn's declared call-ID order rather than
nondeterministic completion order; immediately adjacent result-before-call
inversions are repaired, split assistant text/function items are merged, and
duplicate call/result IDs fail closed. Request-order result copies are not used
for prompt rendering, so equivalent parallel executions reach the same exact
prompt-token and KV-cache identity.

The real added-tool branch also passed the exact-output check: adding one schema
in the middle of a 131-tool Qwen2.5-1.5B catalog reused 8,192 of 17,830 prompt
tokens (8,896-token raw LCP), took 5.15 seconds versus 8.13 seconds on a fresh
cold engine (36.7% less), and emitted the same greedy ID as that cold control.

The corresponding real Qwen2.5-1.5B MXFP4 run changed a 21,234-token random
permutation from a 10.18-second cold prefill to a 0.027-second exact memory hit,
with the same greedy ID. The final path also caches eight exact rendered-prompt
token capsules per resident engine, carries those already validated IDs into
generation, and caches compiled Jinja templates. A second 17,694-token
permutation then measured 5.00 ms end-to-end (3.50 ms prepare + 1.50 ms
generation), versus 7.55 seconds cold; engine re-tokenization was 0.023 ms.
Prompts below 1,024 tokens do not occupy the token-capsule LRU. These are
repeat/catalog reuse results, not a claim that a first cold request became
hundreds of times faster. A Qwen2.5-7B MXFP4 60-tool behavior gate retained
10/10 correct tool names and required argument keys in both request and
canonical order.

This matches the public exact-cache contracts: [OpenAI prompt caching](https://platform.openai.com/docs/guides/prompt-caching)
requires exact prefix matches and identical tools, while [Anthropic prompt
caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
defines the cumulative hierarchy as tools, then system, then messages. It also
explains why normalizing unstable client JSON/tool order is useful before the
model sees it.

An unchanged tool after a changed earlier prefix does not have exact reusable KV:
its hidden state causally depends on the complete preceding token sequence. RoPE
rebasing fixes position phase, not changed attention or residual state. Exact
multi-version caching must therefore use parent-hashed prefix nodes:

```text
node = H(parent || module_token_ids || model/arithmetic identity)
```

Persistent DAG storage additionally needs immutable content-addressed KV blocks,
atomic manifests, checksums, reader leases, and garbage collection. Semantic
modules index the tree; they do not authorize a branch by themselves. This is an
adaptation of [vLLM's parent-hashed automatic prefix cache](https://docs.vllm.ai/en/latest/design/prefix_caching/),
not a claim that arbitrary modules can be reused exactly.

Fast Qwen and OLMoE now also have an experimental position-independent path for
edit-heavy catalogs. The renderer records only token-aligned compact JSON tool
spans; templates whose punctuation crosses a token boundary fail closed to the
ordinary exact-prefix path. After an insertion/removal, the engine first keeps
the best exact chunk prefix, then applies an
[EPIC/LegoLink](https://arxiv.org/abs/2410.15332)-style repair: the first four
tokens of every unchanged tool are recomputed layer-by-layer against the complete
assembled cache, unchanged tails reuse prior KV, and post-RoPE keys are rotated
by their position delta. OLMoE additionally reruns the released router for every
selected position and uses either resident gathered experts or ordinary paged
expert fetches. Uncached tools plus the user/query suffix are fully recomputed.
Exact repeat/endpoint/strict-extension hits always take priority.

Qwen3-VL uses its separate single-owner prompt cache. Its selective sweep uses
the complete three-axis M-RoPE delta, always recomputes media-token positions,
splices exact tower embeddings, and reapplies selected DeepStack contributions.
Image/video suffixes still fail closed; only token-aligned text-tool spans are
detached. The default requires at least 128 avoided token positions plus a
projected duplicate-KV peak below the governor's freshly sampled ceiling.
`VMODEL_FAST_TOOL_PIC=0` disables it; `VMODEL_FAST_TOOL_PIC_REPAIR_TOKENS` and
`VMODEL_FAST_TOOL_PIC_MIN_SAVINGS` expose the quality/work point. Telemetry reports
`prompt_cache_source=tool_pic`, selected/reused/repaired positions, admission,
and PIC wall time.

Measured edited-catalog gates cleared the project threshold by a wide margin. A
24->25-tool Qwen2.5-1.5B MXFP4 insertion preserved all 32 greedy IDs and reduced
exact-prefix prefill from 349 ms to 114 ms (3.06x). Qwen3-4B MXFP4 preserved all
32 IDs at 1.426 s -> 316 ms (4.51x). On Qwen2.5-7B MXFP4, five targets spanning a
60->61-tool insertion retained every tool name, required `path`, and every greedy
ID; median prefill speedup over the exact-prefix baseline was 6.86x at a 4.35 GB
single-request peak. Reproduce with `tests/fixtures/qwen_tool_pic_gate.py`.

The resident expert-MXFP4 OLMoE profile kept the requested tool/path and all 32
greedy IDs while reducing edited prefill from 676 ms to 157 ms (4.32x) at a
4.90 GB peak; its pageable-expert five-case gate had a 4.96x median and matching
streams. Three Qwen3-VL image/tool targets retained every required path and all
144 greedy IDs while moving median prefill from about 499 ms to 131 ms (3.80x)
at a 3.22 GB peak. Reproduce the multimodal result with
`tests/fixtures/qwen3vl_tool_pic_gate.py`.

Capsule identities/spans are part of the immutable checksummed hot-KV checkpoint
manifest. Approximate status also survives, and approximate generations cannot
feed another PIC edit. A real Qwen2.5-1.5B restart restored all 24 spans, exercised
PIC on the first edit, retained all 32 IDs, and reduced prefill 395 ms to 203 ms
(1.95x); see `tests/fixtures/qwen_tool_pic_restart_gate.py`.

This does not make detached KV lossless: cached tails retain their old preceding
context. [CacheBlend](https://arxiv.org/abs/2405.16444) discrepancy selection may
repair harder cross-tool dependencies but adds dynamic work;
[HYPIC](https://arxiv.org/abs/2607.01299) independently reports that the largest
hybrid-stack deviations concentrate at segment beginnings and repairs a small
seam window, consistent with this runtime's boundary policy. For multimodal
reuse, [PRCR](https://arxiv.org/abs/2606.26631) identifies stale positional
binding as the direct-reuse failure and rebinds visual keys, while
[Kamera](https://arxiv.org/abs/2606.23581) adds a low-rank conditioning patch for
harder cross-chunk reasoning. The latter is a possible accuracy lever if future
quality gates expose a gap; it is not free and the current local tool gates
already match exact greedy streams.
[MEPIC](https://arxiv.org/abs/2512.16822) and
[MiniPIC](https://arxiv.org/abs/2606.13126) store unrotated keys and target paged,
concurrent sharing. The established path above still stores post-RoPE keys and
builds a private relocated destination. An experimental dense-Qwen path now uses
an engine-wide, one-token-page pool instead: physical pages store scaled but
unrotated K plus V, per-cache block tables provide logical order, and refcounts
let an edited destination retain unchanged tool pages before its source is
released. The custom Metal attention kernel rotates K at logical read position;
the same physical page can consequently appear at different positions without a
copy or mutation.

The kernel alone is not faster at every context. Local microbenchmarks were
1.42x/1.34x faster for one/four queries at 512 keys, but approached neutral or
slower beyond roughly 2K keys, and wide prefill was substantially slower than MLX
SDPA. The retained hybrid therefore uses the page-table kernel for short decode,
gather+MLX SDPA for selective prefill, and a request-local pre-rotated hot view for
long decode. That hot view is discarded before the state returns to the LRU, so
the durable in-memory representation remains shared pages only.

On the real Qwen2.5-7B MXFP4 60->61-tool/five-case gate, shared pages preserved
all 5/5 established-PIC streams and every requested tool/path, improved edited
prefill by a 1.068x median and the whole edited request by 1.040x, and lowered
peak Metal from 4.773 GB to 4.701 GB. It remained 6.555x faster than exact-prefix
recomputation. The underlying four-token PIC approximation itself matched the
exact-control stream in 4/5 cases; a 12-token seam repaired the remaining stream
but cost substantial prefill work, so the faster four-token quality point remains
unchanged. Qwen3, OLMoE, and small Qwen2 trials preserved the comparison streams
but failed either the speed or stable-quality admission gate; the server therefore
admits shared pages only for 7B-class-or-larger dense Qwen2 when explicitly enabled
with `VMODEL_FAST_TOOL_PIC_SHARED_PAGES=1` (default 0).

This first shared format is engine-local. Durable hot/prompt-KV serialization,
disk spill, GLM/MLA, and multimodal M-RoPE fail closed rather than treating
unrotated physical pages as the existing rotated dense format.

## YaRN long profile

For Qwen2 checkpoints, `lossy-long-<name>` enables static YaRN with factor 2 by
default:

```bash
.venv/bin/python -m runtime.server --port 8077

# Native released window, compact tools, exact hot-prefix reuse
# model: lossy-Qwen2.5-1.5B

# Experimental factor-2 / 65,536-position profile
# model: lossy-long-Qwen2.5-1.5B
```

Override the factor with `VMODEL_FAST_LONG_YARN_FACTOR`. The exact finite factor
is part of engine/cache identity, so native, factor-2, and factor-4 states cannot
cross-reuse. The server rejects non-Qwen2 fast-long profiles.

The [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
specifies a 32,768-token window. [YaRN](https://arxiv.org/abs/2309.00071) beyond
that remains extrapolation, but the local factor-2 MXFP4 gate now passes exact
needle retrieval plus schema-constrained two-tool loops at 32K, 48K, and 64K.
The cold prefills took 18.4s, 49.5s, and 125.4s and peaked at 2.76GB, 3.71GB,
and 4.67GB Metal respectively; the 64K post-tool response emitted `DONE`/EOS
without repetition. Strict next-turn extensions now reuse the complete retained
post-generation endpoint rather than dropping to the previous 4,096-token
boundary: the 32K follow-up fell from 3.39s to 0.48s (7.0x), while arbitrary
edited branches remain chunk-aligned. Reproduce with
`tests/fixtures/qwen_lossy_long_gate.py`. YaRN increases capacity; it does not
reduce a cold request's prefill tokens or attention pairs, and factor 4 remains
ungated.

## Safety and validation

Before any MLX/model run, require swap free at least 2 GB, internal-root free at
least 5 GB, no other MLX job, and projected Metal below the smaller of MLX's
recommended working set and current active Metal plus system-available memory
after the governor reserve. The
server validates `prompt + requested output` against the smaller of the model
window and any runtime correctness bound before sending streaming headers. Zero,
negative, boolean, and fractional token budgets are rejected.

Vision is a separate state owner: it releases prior text/hot KV before allocating
its own cache. Attention is global within one image or temporal video segment
and segmented across video time, matching the official Qwen3-VL `cu_seqlens`
contract. Preflight rejects more than 4,096 spatial patches per segment or more
than 4,096 retained merged media tokens in aggregate.
Exact multimodal prompt-KV hits and text-only suffix extensions now bypass the
vision tower before embedding-cache lookup; media hashes and expanded tokens have
already established identity. The suffix guard checks both image and video token
IDs. With the embedding LRU forced off, an exact local repeat retained its ID,
skipped a measured 25.6 ms tower pass, and completed in 2.95 ms.

Run the dependency-free checks at any time:

```bash
.venv/bin/python tests/test_toolcalls.py
.venv/bin/python tests/test_server_pure.py
.venv/bin/python tests/test_yarn_parameters.py
.venv/bin/python tests/test_vision_positions.py
.venv/bin/python tests/test_incremental_decode.py
```

These dependency-light gates run without downloading a production checkpoint.

Streaming uses cumulative stateful detokenization rather than concatenating
`decode([token])`; the latter corrupts byte-fallback Unicode and can leak partial
stop strings. Chat Completions, Responses, and Anthropic Messages all support
typed image/video streams. JSON/schema output and required/specific tools use
XGrammar constraints rather than advisory prompting. The fast Qwen3-VL resident
text trunk also pipelines decode
with its explicit compressed M-RoPE position; a five-pair 64-token local A/B kept
token IDs identical and improved median decode by 19.7% (125.5 to 150.2 tok/s).
Nested tool history and media sources are validated before model resolution;
remote media I/O and request capture also stay outside the serialized inference
section. Chat streams support `stream_options.include_usage`, while unsupported
stateful/multi-choice controls fail explicitly instead of silently
changing request semantics.

GPT-OSS remains explicitly quarantined for correctness work: its current YaRN
path does not yet match OpenAI's inverse-frequency and `truncate:false` reference.
Telemetry and cache identity label that profile `unvalidated`; do not reuse prior
GPT-OSS token-correctness claims until an official-oracle repair lands.
