# DFlash2 fused convolution and residual projection

These are research-only, default-off proposal optimizations for the official
Qwen3.8 DFlash2 sidecar.  They never change the served Huihui target.  Every
proposal is still checked by the target-authoritative verifier, including
exact rejection correction and Qwen full-attention/DeltaNet state commit.

## Fused dynamic convolution

Enable the one-dispatch Metal draft convolution with:

```bash
export VMODEL_QWEN_DFLASH2_FUSED_DYNAMIC_CONV=1
```

The kernel implements DFlash2's two-tap group-dynamic causal convolution in
one dispatch and serial FP32 accumulation.  When residual projection is
enabled, a second threadgroup form fuses the 5,120-wide direction reduction
and subtraction into the convolution instead of materializing the branch and
launching a separate projection.  Both forms are formula-equivalent but not
BF16 byte-identical to the MLX graph, so they are deliberately confined to the
draft.  The real-geometry microbenchmark is reproducible after a passing
memory preflight:

```bash
.venv/bin/python tests/fixtures/dflash2_dynamic_conv_bench.py \
  --warmup 30 --repetitions 400 \
  --result logs/dflash2_dynamic_conv_bench_20260824.json
```

The plain convolution improved 0.27096ms to 0.19483ms (1.391x).  With
projection, the already-fused convolution plus a separate projection measured
0.24635ms, while the combined kernel measured 0.23508ms (1.048x), saving only
0.225ms over all 20 calls in a five-layer proposal block.

## Rank-one sidecar projection

The artifact builder accepts a small safetensors file of measured F32
directions, sign-aligns and averages only a globally coherent set, normalizes
the result, and atomically writes a target-bound artifact:

```bash
.venv/bin/python -m runtime.dflash2_ablation from-safetensors \
  --input /path/to/refusal_dirs_qwen38.safetensors \
  --output /external/path/dflash2-direction \
  --target-config models/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4/config.json \
  --draft-revision dedf8df68adfb1afeaf7b7480c0a0243108177b4 \
  --source-repository OWNER/REPOSITORY \
  --source-revision PINNED_COMMIT
```

Enable it only with an explicit artifact and strength:

```bash
export VMODEL_QWEN_DFLASH2_ABLATION_DIRECTION=/external/path/dflash2-direction
export VMODEL_QWEN_DFLASH2_ABLATION_STRENGTH=1
```

The runtime fails closed on schema, direction hash, target config hash, draft
revision, hidden width, unit norm, or strength mismatch.  It applies the
projection to both residual-writing branch outputs in every DFlash layer,
before residual addition; it never repeatedly erases the accumulated residual
stream.

The first tested direction came from an Ektome-vs-base rank-one weight-delta
artifact, not from Huihui or from DFlash2-specific contrastive activations.
It preserved exact target tokens/state but did not increase acceptance and
regressed wall time.

The bounded extractor can now recover Huihui's own direction without
downloading another complete checkpoint.  It range-reads only named BF16
residual-writer tensors from a pinned official Qwen revision, compares them to
the local Huihui BF16 checkpoint, hashes every tensor, verifies a retained
pre-ablation layer is byte-identical, and refuses output unless the edited
directions are subtractive, projection-shaped, rank-one, and coherent:

```bash
.venv/bin/python -m runtime.qwen38_rank1_probe \
  --base-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --ablated models/Huihui-Qwen3.8-27B-abliterated \
  --ablated-revision d42ca8978c5a66e92c3446d46e8adfe03ef692ff \
  --target-config models/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4/config.json \
  --draft-revision dedf8df68adfb1afeaf7b7480c0a0243108177b4 \
  --output models/Qwen3.8-27B-DFlash2-ablation-huihui-direction \
  --report logs/qwen38_huihui_rank1_probe_20260824.json
```

The measured layer-16/19/20/63 deltas have 0.9925–0.9944 rank-one energy,
at least 0.99993 projection-form cosine, 0.999996 cross-layer direction
coherence, and mean effective strength 1.29963.  Layer 14 is byte-identical to
official Qwen, matching Huihui's stated first-15-layer retention.  Extraction
took 12.36s and wrote direction fingerprint
`0db65187...39681fe`.

This proves that the actual Huihui edit direction is recoverable; it does not
prove that applying the target's coefficient to a differently trained DFlash
architecture improves proposals.  Two cap-4, eight-token admission attempts
with the Huihui direction were refused by the unchanged governor at the
serial verifier, about 10MB over the safety ceiling.  A third attempt after
fusing projection into the convolution was also refused at essentially the
same active footprint, proving the projection temporary was not the limiting
allocation.  They are recorded as a memory stop, not worked around.  Promotion
still requires a successfully admitted heterogeneous acceptance/quality
corpus, or DFlash2-specific harmful/harmless activation captures.

A bounded three-token oracle did admit.  At measured Huihui strength 1.29963,
the combined kernel emitted `[271, 12, 2972]` and reproduced both visible
output SHA-256 `c9165f...392c` and the complete 129-tensor target endpoint
SHA-256 `979303ca...b4c85` from the unprojected/unfused reference.  Both arms
accepted 1/1 proposal.  Candidate wall was 18.539s versus 17.320s, with 2.656s
versus 0.909s sidecar load time, so it is a correctness witness and a
performance stop.

## Promotion gates

- Keep both switches off by default.
- Require byte-identical greedy target output and complete endpoint hashes.
- Require stochastic `p/q` distribution tests for sampled traffic.
- Require at least 10% cold wall improvement on heterogeneous real request
  shapes, not one pinned capture.
- Refuse any run that crosses the 8.5GB Metal or swap-growth gates.
- Do not lower the 5.3GB system reserve to make a sidecar fit.

## Adaptive DFlash2 to native-MTP fallback (2026-08-25)

The serving adapter now has a target-verified, default-off productivity gate:

```bash
export VMODEL_QWEN_DFLASH2_NATIVE_MTP_FALLBACK=1
export VMODEL_QWEN_DFLASH2_FALLBACK_MIN_ROUNDS=4
export VMODEL_QWEN_DFLASH2_FALLBACK_MIN_ACCEPTED_PER_ROUND=1
```

Only acceptance measured by the authoritative target can trigger the switch.
After a switch, the released-BF16 native MTP sidecar is loaded for one proposal
round and released before the target verifier.  The proposal source therefore
changes efficiency only: target verification and exact stochastic `p/q`
correction continue to define the served distribution.  The adapter also moves
DFlash2's BF16 context K/V to bit-preserving CPU `uint16` storage between the
draft and target sweeps, restoring it only if the next round still uses
DFlash2.  Per-round proposed/accepted counts, draft/verify/context time, source
trace, switch point, sidecar bytes, and context suspend/restore bytes and time
are emitted through the API timing object.

The unmodified 134-tool, 6,339-input-token, max-64 capture completed with this
hybrid path, but it is a performance **STOP**:

| Path | Wall | Decode | Output | Target sweeps | Accepted / proposed | Decode reads | Peak Metal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native BF16 MTP | 201.7076s | 141.4487s | 35 | 19 | 16 / 19 | 264.855GB | 3.906GB |
| DFlash2 then native MTP | 272.4593s | 210.3759s | 42 | 28 | 14 / 36 | 444.630GB | 3.677GB |

The hybrid switched after four DFlash2 rounds (`DDDD` then 24 native-MTP
rounds).  DFlash2 accepted 3/12; the fallback accepted 11/24.  The successful
admission required an unpinned 675MB LM head, which amplified verifier reads;
the run read 460.005GB total versus 278.001GB for the native baseline.  A
subsequent unit-gated fix corrected the fallback draft RoPE position from the
target endpoint to the last committed token.  The table intentionally reports
the measured pre-fix run, not an unmeasured speed claim.  Even perfecting its
fallback acceptance cannot recover the large unpinned-head read penalty, and
the first four DFlash2 rounds were already less productive than native MTP.

Exact paged-KV admission experiments at 128MB and 64MB exposed and fixed two
general correctness gaps: speculative rollback can now truncate individual
spilled layer pages without materializing the whole prefix, and the global
cache offset is the maximum populated layer length during mixed-depth Qwen
prefill.  Those smaller caps did not clear the live DFlash2 memory gate; 64MB
increased transient page reload/materialization pressure.  They remain
explicit opt-ins, while the native-MTP profile stays the selected fast path.
