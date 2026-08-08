# Run 5 — SoftShare-10X strategy pivot

## Decision

The Run-4 hard-recursive architecture is no longer the primary path.

The goal is **10×, not 50×**. Exact recursive sharing spends too little representation capacity on layer specialization and creates correlated depth-wise quantization error. For a 10× target, that is an avoidable quality sacrifice.

Run 5 uses **SoftShare-10X**:

`W_(layer,type) = S_type + A_(layer,type) B_(layer,type)`

- `S_type`: one shared full-rank canonical-Q4 matrix per projection type;
- `A B`: depth-specific low-rank canonical-Q4 residual;
- small layer-specific state retained;
- direct packed inference computes `Sx + A(Bx)` without a dense per-layer reconstruction;
- residual/KV rank and rescue pages are increased until the complete deployment approaches the 10× boundary.

The latent-KV work remains useful, but its rank is chosen for the best quality that still fits 10× rather than maximum cache compression.

## Why this is the higher-probability 10× path

### The 10× budget can afford layer specialization

At Mistral/Llama scale, residual ranks around 96–128 are structurally compatible with the target. There is no reason to force 32 logical layers through one almost-identical physical block when tens of megabytes remain available for depth-specific information.

### Controlled evidence: actual storage codec

A first Run-5 control mistakenly used 128-value grouped Q4 for residual factors while the implemented converter/runtime used canonical `Q4_ROW`. Its paired toy byte/quality figures—including the earlier ~10.14× and ~10.05× toy results—are **revoked**.

The authoritative rerun uses exactly the converter/runtime codec for every A/B factor row. Canonical-Q4 teacher NLL: **1.90547**.

| residual rank | complete tiny-model tensor reduction | final Q4 NLL | ppl ratio vs Q4 teacher |
|---:|---:|---:|---:|
| 3 | **7.099×** | **1.85275** | **0.94864×** |
| 2 | **8.095×** | **1.98593** | **1.08378×** |
| 1 | **8.411×** | **1.91066** | **1.00520×** |

This is not complete-10× evidence. At d=128, `Q4_ROW` scale bytes dominate factors only 1–3 values wide, so this toy geometry is a poor file-size proxy.

It is useful strategy evidence: explicit low-rank layer residuals preserve the controlled teacher's behavior well under the actual codec. Rank3 slightly beats the finite-step teacher after extra recovery/distillation; that must not be interpreted as compression increasing general intelligence.

The scale hypothesis to test is `3/128 = 2.34375%` relative rank → approximately rank96 at hidden4096, where factor scale overhead is much smaller relative to payload.

### Literature direction

Layer-wise LoRA/relaxed recursion, cross-layer basis sharing, and matrix-dictionary sharing independently support retaining layer-specific low-dimensional information. SoftShare itself is not claimed as wholly novel; the project contribution is the 10× budget policy plus packed execution, latent-KV, progressive rescue, and bounded-source-residency conversion in one format/runtime.

## Why not pure extreme quantization?

Moving from a ~4.5-bit Q4-class baseline to ~1.58-bit weights is only about a 2.8× same-parameter reduction. It cannot independently deliver 10×, and post-training extreme quantization creates more quality risk than this goal requires.

## Why not simply distill to a 10× smaller dense model?

That is a legitimate deployment baseline and must be compared. It is not the same representation problem, however: it discards the source model's layer-specific state. Distillation is retained as a recovery objective and as an iso-byte control.

## Real target: Mistral-7B-v0.1

Planner architecture:

- 32 layers
- hidden 4096
- intermediate 14336
- 32 attention heads
- 8 KV heads
- head dimension 128
- vocabulary 32000
- sliding window 4096

Named baseline:

- `mistral-7b-v0.1.Q4_K_M.gguf`
- exact size **4,368,438,912 B**
- SHA-256 `ce6253d2e91adea0c35924b38411b0434fa18fcb90c52980ce68187dbcbbe40c`
- exact integer 10× file ceiling **436,843,891 B**

## Starting budget: rank96 weights / rank64 KV

Resident tensor model:

- weights **369,495,040 B** = **11.8227×** vs exact Q4_K_M;
- 4K latent KV **44,105,728 B** vs **536,870,912 B** FP16 = **12.1724×**;
- weights + 4K KV **11.8600×**;
- equal-common-scratch headroom before that tensor ratio reaches 10×: **85,478,016 B**.

Complete-file planning additionally charges the current `.larc` manifest, 522 page records, Q4 page headers, 4 KiB alignment, plus a conservative 4 MiB tokenizer/config reserve:

- serialized weight `.larc`: **371,302,608 B**;
- with auxiliary reserve: **375,496,912 B**;
- conservative file ratio **11.6338×**;
- remaining bytes before the exact 10× file ceiling: **61,346,979 B**.

That ~61 MB is quality budget. The converter should spend it only where real held-out validation shows the largest gain/byte.

The overhead-aware sweep also rejects rank144: tensor payload alone appears barely above 10×, but container + auxiliary reserve gives only **9.964×**.

**Mistral quality is completely unvalidated.**

## Runtime

`runtime/larc_q4.cpp` now directly evaluates packed `Sx + A(Bx)`. Native L1 correctness against separately dequantized S/A/B has max absolute error **9.54e-7**, rank-sized scratch, and no dense W reconstruction.

## Bounded-source-residency conversion

The converter does not require a complete source checkpoint to be stored locally.

### Input

`larc/safetensors_range.py` reads individual tensors from local shards or exact remote ranges. Remote responses must be HTTP `206` with `Content-Range`; a server that ignores Range is rejected instead of silently transferring a full multi-GB shard.

### Output

`LARCv2StreamWriter` reserves the manifest/page table, writes each compressed payload immediately, keeps only page records, and patches metadata at finalize.

### SoftShare conversion

Per matrix family:

1. stream source layers and accumulate shared mean;
2. quantize the mean to canonical Q4;
3. discard the float precursor;
4. reconstruct the **stored/dequantized** shared base;
5. stream each source layer again;
6. fit its low-rank residual against that stored base;
7. quantize/write A and B immediately;
8. release source/SVD state before continuing.

A local synthetic integration test with two actual SafeTensors shards passed: **42 expected/actual pages** and every `.larc` CRC verified. This validates the mechanism, not real Mistral conversion.

## Hard gates

Run 5 is not a real 10× success until:

- [ ] real pretrained Mistral conversion (L3)
- [ ] same-token perplexity/tasks/generation versus the named Q4_K_M baseline
- [ ] actual self-contained output file ≤ **436,843,891 B**
- [x] direct packed SoftShare CPU operator correctness (L1)
- [ ] real conversion demonstrates bounded source residency
- [ ] realistic-context KV quality
- [ ] measured same-context process/device peak memory (L4)
- [ ] comparison against a genuinely ~10× smaller dense/distilled model and other feasible iso-byte alternatives
