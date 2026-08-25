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
one dispatch and serial FP32 accumulation.  It is formula-equivalent but not
BF16 byte-identical to the MLX graph, so it is deliberately confined to the
draft.  The real-geometry microbenchmark is reproducible after a passing
memory preflight:

```bash
.venv/bin/python tests/fixtures/dflash2_dynamic_conv_bench.py \
  --warmup 30 --repetitions 400 \
  --result logs/dflash2_dynamic_conv_bench_20260824.json
```

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

The locally tested direction came from an Ektome-vs-base rank-one weight-delta
artifact, not from Huihui or from DFlash2-specific contrastive activations.
Consequently it is not advertised as a Huihui sidecar ablation.  The first
real gate preserved exact target tokens/state but did not increase acceptance
and regressed wall time.  A genuinely useful follow-up requires either the
original Huihui direction or DFlash2-specific harmful/harmless activation
captures plus a heterogeneous held-out acceptance/quality corpus.

## Promotion gates

- Keep both switches off by default.
- Require byte-identical greedy target output and complete endpoint hashes.
- Require stochastic `p/q` distribution tests for sampled traffic.
- Require at least 10% cold wall improvement on heterogeneous real request
  shapes, not one pinned capture.
- Refuse any run that crosses the 8.5GB Metal or swap-growth gates.
- Do not lower the 5.3GB system reserve to make a sidecar fit.
