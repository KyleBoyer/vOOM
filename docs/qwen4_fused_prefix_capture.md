# Qwen4 cold aligned-prefix capture: integration and proof contract

As of2026-09-06 the helper is wired into an **explicit, default-off**
`...hot-kv-aligned-compact-fused` profile;866 regressions pass. Independent
real full/prefix state gates and the focused completed cold/repeat raw-token
gate now pass; overall pressure gates still FAIL. Broader real-request,
extension, full harness and Plex acceptance remain open. The objective is to avoid a separate cold
prefix sweep while retaining the existing aligned1024-token hot checkpoint.
The focused1611-token workload currently reads an additional101,039,536,440
bytes for that split. The new focused candidate removes those reads and took
357.266s cold versus the historical compact control443.4725s. That modified
two-tool comparison is not an achieved full-harness latency target. See the
current STATUS entry for scope, pressure failures and complete receipts.

The next varied-domain control (source439faf7) exposed a quality failure before
the fused comparison: the modified two-original-Plex-tool/no-developer/nonstream
case completed146 tokens in633.0516s but produced two search calls where exactly
one was required. Both call-count and pressure gates FAIL. No fused arm ran;
this is not evidence of a fused regression or generalized success. Diagnose
raw-generation versus parser origin on that frozen request before proceeding;
do not tune the prompt to conceal it. The replay gate now binds name/argument
checks and accepts `--expected-function-call-count 1`; canonical per-call hashes
will distinguish identical calls from repeated names without persisting text.

Follow-up on source6e3a0d4 confirms both identical search calls already exist
as two disjoint Hermes frames in the raw generated text. Their canonical hashes
match both protocol calls and the expected arguments; all146 emitted IDs and
the entire raw text match the previous failed control. New diagnostic costs
0.542ms,541 regressions pass. The635.2309s request still fails call count and
memory gates. No fused comparison or speed/quality gain was established.
Next compare native MTP with ordinary greedy target generation on that frozen
case; raw-origin attribution alone does not prove the model, rather than its
speculative decoder, is responsible. No host deduplication or prompt retuning.

That comparison is now complete on source `ca753cd`: the explicit `...compact-ar`
profile changes only native MTP depth 3 to 0, and ordinary greedy generation
matches the ENTIRE native-MTP 146-token/raw-490B witness plus both raw frames and
canonical parsed calls. Native MTP did not introduce the duplication on this
frozen request; ordinary target generation in this runtime produces it too.
This is not an independent official-BF16 oracle or a broad quality proof.
The modified two-Plex-tool/no-developer/nonstream/max-512 call took 1227.8277s
HTTP / 1010.4818s decode, versus historical native-MTP 635.2309s / 419.6825s.
Ordinary decode reads 1,406,363,212,800B versus 658,745,963,712B; prefill bytes
are unchanged. Head lifetime differs: ordinary retains the 1.271GB head through
single-token trunks/idle, while native MTP suspends it around multi-token
verification and releases it at request end. Do not call this an equal-lifetime
speed/pressure A/B or attribute the entire wall ratio to acceptance alone.
Both call-count and pressure acceptance still FAIL. Ordinary peak Metal 7.726GB
stays below 8.5GB but used swap grows 2.484GB and swap-out 61.587MB; final 8.540GB
available does not erase the churn. Parent exit 1/no timeout or source drift;
all jobs ended. 138 pure regressions pass in 2.16s. Private
`logs/qwen4_media_ar512_20260906.json` and matching gate/server artifacts.
This isolation gate is finished; next add correctly timed phase-head pressure
observations, then measure a bounded exact lifetime/reservation improvement.
Do not repeat this same control or conceal the known quality failure.

## Proven helper scope

`runtime.qwen4_prefix_capture.AlignedPrefixCapture` requires exact concrete
KV/KDA/Qwen4 companion types, independent complete lists and genuinely empty
position-zero state. It supports only plain, uncompressed, unwindowed RAM
caches; no DSA companion, KDA spilling/factor capture, or derived QSA pool.
The prefix is positive, strictly interior and a multiple of the original
fixed tile. Geometry is copied from config when the builder is constructed.

The helper imports no model or MLX code. Supply the concrete cache types,
BF16 activation / FP32 recurrence / int32 position dtypes, `mx.eval`, the
existing `copy_history_bits`, governor scratch reservation and a factory that
creates a new empty KVCache with fresh KDAStateCache/Qwen4ExpStateCache owners.
The caller must already charge the additional retained logical state and
backing allowance. Scratch reservation is per layer, twice the small history
payload, without credit for future release of the source's padded backings.

`begin_sweep` rechecks a cold, unchanged source and exact offset/total/tile.
`observe_tile` must see EVERY unchanged post-attention tile in original
layer-major order, including all prefix/suffix tiles. It validates local
geometry each time; global KV offset is deliberately not used while layers
cover different positions. At the prefix, it evaluates all shared arrays
(including below-indexer-threshold raw QSA state) and copies only KDA/PLE
convolution histories. Source arrays/list entries are not replaced.

Final-tile object witnesses protect non-position-indexed KDA state against
same-shape rewinds before `finish`. `finish` requires all layers and tiles,
revalidates full source state and creates/populates a new destination. It
returns `(prefix_cache, scalar_stats)` exactly once. On failure it drops private
records and clears only a previously validated, independently owned partial
destination. Never clear an invalid factory result that could alias the source.

Synthetic tests cover4/8/1024-token boundaries, BF16/F16 bit preservation,
all relevant state/metadata, source ownership, real depthwise-convolution
continuation, invalid geometry/order, interruptions and partial publication.
They do not prove full-model arithmetic/scheduling or serving pressure.

## Engine hookup contract (implemented; retain these invariants)

1. Add a default-off runtime/YAML/server identity flag and an explicit profile
   extending `qwen38-flash-next-uncensored-fp8-exact-pipeline-hot-kv-aligned-compact`.
   Do not change any existing profile or automatic behavior. Keep the existing
   retained logical/QSA-view admission charge; do not double-charge it or
   assume shared arrays prove a full physical-memory bound.
2. In `generate`, use a separate deferred Qwen4 prefix-token scalar. The private
   builder is created only inside the complete sweep's owning wrapper. Cold gating
   needs aligned+compact+hot eligibility, `pos==matched==0`, fixed layer-major
   tiles, plain RAM KV, no approximate state, persistence, adaptive paging,
   adaptive chunks, checkpoints or last-token-separate path. Unsupported
   matches/extensions keep the existing split path, not reinterpret
   their recurrence. A builder that loses eligibility must fail closed before
   any alternate sweep.
3. Defer BOTH the prefix mini-sweep and its later fork/`pos=stable_boundary`
   bookkeeping. The existing `if not fuse_qwen_boundary_scaffold` block needs
   an additional Qwen4 exclusion. Leave `pos=0`, `boundary_fork_kv=None` and
   `boundary_fork_tokens=0` until successful completion. Do not reuse the dense
   Qwen3.5 deferral variable or compact a mixed/empty cache.
4. Give `_layer_stationary_qwen4_sweep` an optional builder. Validate its sweep
   contract before fetching weights. Call `observe_tile` just AFTER the
   existing `mx.eval(post_attention)` and attention-time accounting, but BEFORE
   the routing timer reset, route/GEMM work, host overwrite and next tile.
   Do not change `range/end`, tile shapes, records, expert unions, row ordering
   or accumulator evaluation shapes. Record capture time separately and call
   the existing cap/true-peak sampler after each capture.
5. Only after the entire sweep AND final hidden restoration succeed may the
   caller invoke `finish`, set the stable boundary cache/token count and merge
   typed scalar telemetry. Do not run whole-fork compaction again: histories
   are already compact. On any exception/client interruption outside helper
   callbacks, explicitly `abort` and drop the builder. No partial cache may
   reach `last_kv`, hot slots or persistence. Matched-prefix repeats retain the
   existing proven complete-fork/compactor path.
6. Add actual call-site/flag/profile/admission/cleanup regressions and run the
   combined suite under a new fresh30-second preflight and supervised gate.

## Real-model proof order (first two focused stages completed)

Keep every result explicit about the modified two-real-tool / short history /
developer / streaming / greedy-seed64001 request. It is not the unmodified
134-tool capture or Plex quality proof. No synthetic replacement of state or
answer text is acceptable in a serving benchmark.

The first fresh max1 endpoint diagnostic (not complete-answer latency) matched
against BOTH independent state oracles:

- Full1611-token endpoint, hidden BF16 bits, actual prepared/generated IDs and
  raw text: `logs/qwen4_boundary_nohot_first_20260905.json` and the completed
  aligned-compact max1 control. Full state SHA
  `ef35e7f30d7a7bd28b956c9d352164a8d89659905c7df288323d89a9b6b7ac77`.
- Retained1024-token prefix: independently computed separate prefix sweep in
  `logs/qwen4_retained_allconv_memory_first_20260906.json`, diagnostic
  `fork_before`/`fork_after`. Full state SHA
  `8842fff6d80f36ce58fb03fb7d3f1a51f78bc639c4ef57471aafc71ad7fa78fe`,
  supplemental starts/windows/pool metadata SHA
  `fd4c4aefb4703727e67e08030a6c799b50d9c2e454e2de2ff2705b5b79c10c3c`.

Use a read-only observer for those retained-state hashes. Do not invoke the
older post-generation detachment diagnostic or call its pressure numbers an
uninterrupted serving pass. Check all component hashes, metadata and history
coverage, not just first-token identity or parsed tool arguments.

The sufficient-output fused cold/repeat run with generation witnesses also
matched both historical aligned controls (see STATUS). Next use fresh unfused
controls and varied-domain/actual-extension candidates with generation witnesses,
no endpoint hashing/mutation or added allocator clearing: cold + identical
repeat, then actual extension, varied domains/tool shapes, unmodified capture
and Plex. The existing modified max512 workspace call naturally completes at
70 tokens; raw ID hash
`381c8319f7f04acfd719eb922593657e5e688b45f0273eb678ae6e9735a9ae9b`.
Record output completion, actual IDs/text, cache reuse, I/O, cold loading,
first-token/prefill/decode/HTTP wall, peak Metal and unchanged pressure gates.
Failure of any gate stays visible; no automatic promotion without broad proof.
