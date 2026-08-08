# Run 5 — SoftShare-10X strategy pivot

## Decision

The Run-4 architecture is no longer the primary path to the project goal.

The goal for this project is **10×**, not 50×. Hard recursive sharing (one physical block repeatedly applied at every depth) spends too little representation capacity on layer specialization. Run 4 proved that this creates strongly correlated quantization error and forces recovery training to repair an unnecessarily severe bottleneck.

Run 5 therefore pivots to **SoftShare-10X**:

`W_(layer,type) = S_type + A_(layer,type) B_(layer,type)`

where:

- `S_type` is one shared full-rank Q4 matrix for a projection type;
- `A B` is a depth-specific low-rank Q4 residual;
- small norms/scalars remain depth-specific;
- inference evaluates `Sx + A(Bx)` directly from packed weights and never reconstructs the full per-layer matrix;
- ranks are allocated under a hard 10× byte budget rather than minimized for their own sake.

The latent-KV work remains useful, but its rank should also be chosen to meet 10× rather than maximally compress the cache.

## Why this is now the highest-probability path

### 1. The 10× budget can afford layer specialization

For a 32-layer, 4096-hidden Mistral/Llama-class model, depth-specific residual ranks around 96–128 are feasible while remaining near or above the 10× Q4-class file-size target. This is qualitatively different from the controlled d=128 model, where the same relative rank is only 3–4.

### 2. Controlled evidence favors soft sharing

A Run-5 probe trained the same 16-independent-block character-LM teacher used for the controlled research program, then replaced only its large matrices with one shared mean plus depth-specific low-rank residuals.

At rank 3 (3.515625% residual parameters relative to one full block; ideal same-bit large-matrix ratio 10.24×):

- teacher FP32 NLL: 1.63122
- raw SVD SoftShare NLL: 3.01887
- after 80 compressed-parameter recovery steps: 1.77953
- Q4 teacher NLL: 1.90547
- quantized SoftShare before Q4 recovery: 2.16378
- after 50 Q4-constrained recovery steps: 1.98021
- final delta vs Q4 teacher: +0.07474 nats/char
- perplexity ratio: 1.07760×

Hard sharing of the same large matrices, even while retaining exact layer-specific 1-D state, produced ~50.84 NLL before recovery. The residual representation therefore preserves substantially more pretrained information per recovery step.

This is controlled/synthetic evidence only; it is not a real-model quality result.

### 3. Literature independently supports the direction

Relevant prior work includes:

- **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** (ICLR 2025): converting pretrained Transformers to shared layers works substantially better when depth-specific low-rank adapters are retained.
- **Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression** (ICLR 2025): cross-layer shared bases plus unique coefficients outperform ordinary SVD compression at large compression ratios.
- **Share Your Attention / MASA** (AAAI 2026): matrix-dictionary sharing across Transformer depth preserves performance while reducing attention parameters.

SoftShare-10X is not a claim of novelty over those papers. LARC's research contribution is the combination of an explicit 10× deployment budget, packed-domain execution, progressive rescue pages, latent-KV compression, and streaming conversion into one runtime/file standard.

## Why not pivot to pure 1.58-bit / extreme quantization?

Native BitNet-style 1.58-bit training is important, but 1.58-bit weights are only about 2.8× smaller than a ~4.5-bit Q4-class representation at equal parameter count. It cannot by itself reach the 10× target. Post-training conversion to such low bit widths is also substantially harder than native low-bit pretraining.

## Why not simply distill to a 10× smaller dense model?

That is a valid deployment strategy but changes the problem from representation compression into model replacement. It also discards the opportunity to preserve the original model's depth-specific learned state. Distillation remains a recovery objective, not the primary representation.

## Reference target for Run 5

Initial real-model target: **Mistral-7B-v0.1**.

Public architecture:

- 32 layers
- hidden size 4096
- intermediate size 14336
- 32 attention heads
- 8 KV heads
- head dimension 128
- vocabulary 32000
- sliding window 4096

Public Q4_K_M GGUF reference size: approximately **4.37 GB**.

The exact file-size baseline must be replaced with exact bytes when an accessible artifact is available; the public 4.37 GB value is rounded and all current Mistral byte ratios are therefore planning estimates.

## Run-5 budget policy

1. Build a **core** comfortably smaller than the 10× ceiling.
2. Measure validation sensitivity per layer/matrix.
3. Spend remaining bytes on rank increments or sparse residual rescue pages in descending validation-gain/byte order.
4. Stop when the serialized file or configured resident-memory budget reaches the 10× boundary.
5. Do not pursue extra compression merely because it is available.

Current recommended starting point:

- weight residual rank: **96** uniform seed, then adaptive allocation;
- latent-KV rank: **64** initially (real head dimension 128);
- shared/residual weights: canonical Q4_ROW;
- latent coefficients: Q2;
- latent min/scale metadata: E4M3-FN;
- K/V basis metrics: FP16 inverse-Gram.

A higher-capacity candidate uses weight rank 128 and KV rank 72 and still remains structurally above 10× before common runtime scratch.

## Conversion architecture

SoftShare supports the original requirement that the entire source model need not exist locally at once.

Two-pass streaming conversion:

**Pass 1**
- stream one source tensor/layer at a time;
- accumulate the shared base for each matrix type online;
- copy/quantize non-shared tensors directly to output;
- discard the source shard/tensor.

**Pass 2**
- stream each layer matrix again;
- subtract the finalized shared base;
- compute the budgeted low-rank residual;
- quantize/write factors immediately;
- discard source data.

Only one source tensor/shard plus shared bases/SVD workspace must be resident. A production converter should support HTTP-range/shard streaming so the original multi-gigabyte model is never stored as one local file.

## Hard gates

Run 5 does not count as complete until:

- [ ] real pretrained model conversion (L3);
- [ ] same-token perplexity/benchmark comparison against the named Q4_K_M baseline;
- [ ] output file <= 10% of exact baseline bytes;
- [ ] direct packed SoftShare operator correctness;
- [ ] measured process/device peak memory at the stated context;
- [ ] long-context quality with the selected KV rank;
- [ ] streaming converter proves bounded local source residency.
