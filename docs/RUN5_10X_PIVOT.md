# Run 5 — SoftShare-10X strategy pivot

## Decision

The Run-4 architecture is no longer the primary path to the project goal.

The goal for this project is **10×, not 50×**. Hard recursive sharing—one physical block repeatedly applied at every depth—spends too little representation capacity on layer specialization. Run 4 showed that this creates strongly correlated quantization error and forces recovery training to repair an unnecessarily severe bottleneck.

Run 5 therefore pivots to **SoftShare-10X**:

`W_(layer,type) = S_type + A_(layer,type) B_(layer,type)`

where:

- `S_type` is one shared full-rank Q4 matrix for a projection type;
- `A B` is a depth-specific low-rank Q4 residual;
- small norms/scalars remain depth-specific;
- inference evaluates `Sx + A(Bx)` directly from packed weights and never reconstructs the full per-layer matrix;
- ranks are allocated under a hard 10× byte budget rather than minimized for their own sake.

The latent-KV work remains useful, but its rank is now chosen to satisfy 10× while preserving quality rather than to maximize cache compression.

## Why this is now the highest-probability path

### 1. The 10× budget can afford substantial layer specialization

For a 32-layer, 4096-hidden Mistral/Llama-class model, depth-specific residual ranks around 96–128 remain structurally compatible with the 10× target. This is qualitatively different from the controlled d=128 model, where a 2.34% relative rank is only rank 3.

### 2. Controlled evidence favors soft sharing over hard recursion

A Run-5 probe trained the same 16-independent-block character-LM teacher used for the controlled research program, then replaced its large matrices with one shared mean plus depth-specific low-rank residuals.

At uniform rank 3/128 = **2.34375% relative rank**:

- teacher FP32 NLL: 1.63122
- raw SVD SoftShare NLL: 3.01887
- after compressed-parameter recovery: 1.77953
- canonical-Q4 teacher NLL: 1.90547
- quantized SoftShare before Q4 recovery: 2.16378
- after 50 Q4-constrained recovery steps: 1.98021
- final delta vs Q4 teacher: +0.07474 nats/char
- perplexity ratio: **1.07760×**

This point is **9.106× for the complete tiny model**, not 10×, because embeddings/position weights/small state and low-rank metadata are disproportionately large at d=128. It is therefore mechanism evidence, not a 10× success claim.

The exact complete-model tiny 10× stress test is materially worse:

- uniform rank 2: **10.142×**, perplexity ratio **1.484×**;
- validation-allocated `qkv=2, o=1, fc1=3, fc2=2`: **10.046×**, perplexity ratio **1.401×**.

These exact-toy-10× points are not acceptable target quality. Their purpose is to prevent the project from conflating dominant-matrix compression with complete-model compression.

Hard sharing of the same large matrices, even while retaining layer-specific small state, produced roughly 50.8 NLL before recovery. Soft residuals therefore preserve far more pretrained depth-specific information.

### 3. The literature independently supports the direction

Relevant prior work includes Relaxed Recursive Transformers with layer-wise LoRA, Basis Sharing across layers, and matrix-dictionary sharing across Transformer depth. SoftShare-10X is not a claim that cross-layer low-rank sharing itself is novel. LARC's research contribution is the explicit deployment byte budget plus packed execution, progressive rescue pages, latent-KV compression, and bounded-source-residency conversion in one runtime/file system.

## Why not pure extreme quantization?

A 1.58-bit representation is only roughly 2.8× smaller than a ~4.5-bit Q4-class representation at equal parameter count. It cannot by itself reach 10×. Extremely low-bit post-training conversion also introduces a much harder quality problem than necessary for a 10× target.

## Why not simply distill to a 10× smaller dense model?

That is a valid deployment strategy but changes the problem from representation compression into model replacement and discards much of the source model's depth-specific state. Distillation remains useful as a recovery objective rather than the primary storage representation.

## Reference target for Run 5

Initial real-model target: **Mistral-7B-v0.1**.

Architecture used by the planner:

- 32 layers
- hidden size 4096
- intermediate size 14336
- 32 attention heads
- 8 KV heads
- head dimension 128
- vocabulary 32000
- sliding window 4096

Named deployment baseline:

- `mistral-7b-v0.1.Q4_K_M.gguf`
- exact file size: **4,368,438,912 bytes**
- SHA-256: `ce6253d2e91adea0c35924b38411b0434fa18fcb90c52980ce68187dbcbbe40c`
- exact size obtained from the Hugging Face raw LFS pointer.

### Structural 10× budget

Recommended starting core:

- weight residual rank: **96**;
- latent-KV rank: **64**;
- modeled weight bytes: **369,495,040** = **11.8227×** smaller than the exact Q4_K_M file;
- modeled compressed KV at context/sliding-window 4096: **44,105,728 B** vs **536,870,912 B** FP16 = **12.1724×**;
- weights + 4096-token KV: **11.8600×**;
- if baseline and LARC incur the same extra common scratch, approximately **85,478,016 B** may be added to each before this structural ratio reaches exactly 10×.

A higher-capacity candidate uses weight rank 128 and KV rank 72:

- weights: **10.6171×** vs exact Q4_K_M;
- weights + 4K KV: **10.6374×**;
- equal-common-scratch headroom: **32,660,096 B**.

These are exact arithmetic for the stated representations and exact Q4_K_M file baseline, but **Mistral quality is completely unvalidated**.

## Run-5 budget policy

1. Build a core comfortably smaller than the 10× ceiling.
2. Measure validation sensitivity per layer/matrix.
3. Spend remaining bytes on rank increments or residual rescue pages in descending validation-gain/byte order.
4. Stop when the complete serialized/resident configuration reaches the 10× boundary.
5. Do not pursue extra compression merely because it is possible.

The initial rank-96 core is deliberately below the budget so real-model validation can decide where the remaining bytes are most valuable.

## Conversion architecture

SoftShare preserves the original requirement that the entire source model need not exist locally at once.

The implemented converter works per matrix family:

**Pass 1**
- read one source tensor at a time from local SafeTensors or an HTTP byte range;
- accumulate the shared mean;
- quantize the finalized shared base to canonical Q4;
- discard the float precursor;
- reconstruct only the stored/dequantized Q4 base used as the residual reference.

**Pass 2**
- stream each source layer tensor again;
- subtract the stored/dequantized shared base;
- compute the budgeted low-rank residual;
- quantize/write A and B immediately into the streaming `.larc` writer;
- discard tensor and SVD workspace before the next layer.

The `.larc` streaming writer retains page records but does not retain all payload pages. The SafeTensors remote reader requires HTTP `206` plus `Content-Range`; it aborts rather than silently consuming a server response that ignored `Range`.

## Hard gates

Run 5 does not count as a real 10× success until:

- [ ] real pretrained Mistral conversion (L3);
- [ ] same-token perplexity/task comparison against the named Q4_K_M baseline;
- [ ] complete output file <= 436,843,891 bytes (10% of exact baseline, subject to integer file-size accounting);
- [x] direct packed SoftShare operator correctness (L1);
- [ ] measured process/device peak memory at a stated context (L4);
- [ ] long-context quality with selected KV rank;
- [ ] streaming converter demonstrates bounded source residency end to end on a real checkpoint.
