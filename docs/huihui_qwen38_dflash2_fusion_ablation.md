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
