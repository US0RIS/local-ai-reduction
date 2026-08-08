# LARC Action Sheet

Canonical technical record for **LARC — Local Adaptive Representation & Compute**. Historical Run 1–3 detail remains in `docs/RUN3_AUDIT_CORRECTIONS.md`; Run-4 hard-recursion evidence remains reproducible but is now a reference/ablation. Artifact authority is `benchmarks/INDEX.json`. Current machine-readable status is `benchmarks/RUN5_STATUS.json`.

## Objective

The project goal in this line of work is **10×**, not 20–50× and not “as small as possible.”

The acceptance target is:

> A real pretrained model represented and executed with **no more than 10% of the named Q4-class baseline's relevant deployment bytes at the same context**, while retaining reasonable capability.

Claims must separately identify serialized bytes, resident weights, KV cache, scratch/workspace, measured process/device memory, context length, quality, throughput, and evidence level.

Evidence levels: **L0** format, **L1** operator/runtime, **L2** controlled trained model, **L2C** controlled post-training conversion, **L3** independent pretrained model, **L4** measured hardware.

---

# Strategic reassessment after Run 4

## What Run 4 proved

Run 4 closed several important methodological problems:

- quality paths must execute the same Q4 representation charged by memory accounting;
- both quantized K and V latent bases require inverse-Gram correction;
- calibration/selection/evaluation streams must be disjoint;
- context length must be attached to every total-memory claim;
- direct packed KV execution must avoid a decoded `T×rank` history;
- benchmark artifacts need committed generators.

The best hard-recursive controlled result remains:

- context 64 synthetic character LM;
- Q4 teacher NLL 1.88548;
- hard-shared + latent-Q2/E4M3 NLL 1.97525;
- perplexity ratio **1.09392×**;
- composed L2C+L1 modeled tensor ratio 12.04× at context 64.

But the same program also exposed the central weakness of hard recursion: one Q4 block reused through depth accumulates highly correlated error. Before Q4-aware recovery the shared model's perplexity was about 2.19× the Q4 teacher.

## Strategy decision

**Hard recursive sharing is no longer the primary architecture.**

It optimizes for a more extreme compression regime than the actual 10× goal and throws away layer specialization that the byte budget can afford to retain.

Run 5 therefore uses **SoftShare-10X**:

`W_(layer,type) = S_type + A_(layer,type) B_(layer,type)`

- one full-rank canonical-Q4 shared base `S_type` per matrix type;
- explicit depth-specific low-rank Q4 residual factors;
- layer-specific small state retained;
- direct packed execution computes `Sx + A(Bx)`;
- ranks/rescue pages consume the available byte budget until, but not beyond, the 10× boundary.

Latent-Q2 KV remains part of the design, but its rank is increased as far as the 10× budget permits rather than minimized.

---

# Run 5 — SoftShare-10X

## Controlled strategy-selection evidence — L2C

Controlled model: d=128, H=4, FF=256, 16 independent teacher layers, context 64, vocabulary 37. Teacher and compressed paths are evaluated after canonical Q4 projection.

Teacher canonical-Q4 NLL: **1.90547**.

| profile | complete-model reduction | final Q4 NLL | ppl ratio vs Q4 teacher | interpretation |
|---|---:|---:|---:|---|
| uniform residual rank 3 | **9.106×** | **1.98021** | **1.07760×** | quality-favorable scale-normalized mechanism point; not complete-model 10× |
| uniform residual rank 2 | **10.142×** | **2.30033** | **1.48417×** | exact tiny-model 10× stress point; quality inadequate |
| adaptive `qkv2/o1/fc1=3/fc2=2` | **10.046×** | **2.24237** | **1.40059×** | better exact-budget frontier, still inadequate |

Artifact: `benchmarks/run5_softshare_control.json`; generator: `tools/run5_softshare_control.py`.

### Interpretation

The tiny model is a deliberately adverse complete-model 10× test: embeddings, positional weights, small state, and factor metadata consume a disproportionate fraction of its bytes. The exact 10× points do **not** meet the intended quality standard.

The important architecture-selection signal is the rank-3 point:

`3 / 128 = 2.34375%` relative rank.

It retains far more pretrained information than hard sharing and reaches perplexity ×1.0776 after Q4-constrained recovery. On a d=4096 model, the same relative rank is rank 96, where fixed overhead is much better amortized.

This scale argument is a hypothesis to test on the real model, not evidence that Mistral quality will match the toy model.

## Direct packed SoftShare operator — L1

Implemented in `runtime/larc_q4.{h,cpp}`:

`y = Sx + A(Bx)`

where S, A, and B remain canonical packed Q4. Scratch is rank-sized and no full `W=S+AB` matrix is reconstructed.

Native test (`tests/native_q4_softshare.cpp`):

- shape: 173×211, residual rank 23;
- max absolute error vs separately dequantized reference: **9.53674316e-7**;
- rank scratch: **92 B**;
- dense reconstruction: none.

Artifact: `benchmarks/run5_native_q4_softshare.json`.

## Named real target and exact baseline

Initial L3 target: **Mistral-7B-v0.1**.

Planner architecture:

- 32 layers;
- hidden 4096;
- intermediate 14336;
- 32 attention heads;
- 8 KV heads;
- head dimension 128;
- vocabulary 32000;
- sliding window 4096.

Named baseline:

- `mistral-7b-v0.1.Q4_K_M.gguf`;
- exact bytes: **4,368,438,912**;
- SHA-256: `ce6253d2e91adea0c35924b38411b0434fa18fcb90c52980ce68187dbcbbe40c`.

## Mistral structural budget

### Recommended core: weight rank 96, KV rank 64

- weights: **369,495,040 B** = **11.82273×** smaller than exact Q4_K_M;
- compressed 4096-token KV: **44,105,728 B** vs **536,870,912 B** FP16 = **12.17236×**;
- weights + 4K KV ratio: **11.86001×**;
- if both baseline and LARC incur the same additional common scratch, **85,478,016 B** can be added to each before the structural ratio reaches 10×.

Weight breakdown:

- embedding + LM head: 131,200,000 B;
- seven shared Q4 matrix bases: 109,137,920 B;
- all rank-96 layer residual factors: 128,624,640 B;
- FP16 norms: 532,480 B.

### Higher-capacity candidate: weight rank 128, KV rank 72

- weights: **10.61712×** vs exact Q4_K_M;
- weights + 4K KV: **10.63743×**;
- equal-common-scratch headroom: **32,660,096 B**.

Artifact: `benchmarks/run5_mistral7b_budget.json`; generator: `tools/run5_budget_planner.py`.

**These are exact byte calculations for the stated representations and exact baseline, but Mistral quality is unvalidated.**

## 10× budget policy

The design no longer rewards compression beyond the goal.

1. Start from a core safely below the 10× ceiling.
2. Measure validation loss sensitivity by layer and matrix.
3. Spend remaining bytes on rank increments/residual rescue pages with the highest validation gain per byte.
4. Recompute the complete serialized/resident budget after every allocation.
5. Stop spending only when the 10× boundary is reached or no additional capacity improves quality.

The rank-96 core is therefore a **starting point**, not an intended final compression ratio. Its extra ~1.86× margin beyond the goal is quality budget.

---

# Bounded-source-residency conversion

The original requirement remains: conversion must not require the complete multi-gigabyte source checkpoint to exist locally at once.

## Streaming `.larc` writer

`LARCv2StreamWriter` now:

- reserves the manifest/page table;
- writes each compressed payload immediately;
- retains only fixed-size page records;
- patches page table/header at finalize.

Writer memory is O(page metadata + current payload), not O(output file size).

## SafeTensors tensor-range source

`larc/safetensors_range.py`:

- parses local or remote SafeTensors headers;
- addresses tensors by exact byte range;
- supports sharded `model.safetensors.index.json`;
- remote sources must return HTTP 206 plus `Content-Range`;
- a server that ignores Range is rejected rather than silently downloading a complete shard.

## Two-pass SoftShare converter

`tools/stream_softshare_convert.py` works one matrix family at a time:

1. stream one layer tensor at a time and accumulate the shared mean;
2. quantize the mean to canonical Q4;
3. discard the float precursor;
4. dequantize the **stored** shared base;
5. stream layers again;
6. fit each low-rank residual against the stored/dequantized shared base;
7. quantize/write residual factors immediately;
8. release source tensor and SVD workspace.

This preserves the representation-consistency rule learned in Runs 3–4.

Synthetic end-to-end test `tests/test_stream_softshare_convert.py` creates two actual SafeTensors shards, converts a two-layer Mistral-shaped model, verifies the expected 42 `.larc` pages and all CRCs, and checks the bounded-source-residency report. CI execution remains pending until a GitHub runner is allocated.

---

# Current claim boundary

The strongest defensible statement after the Run-5 strategy reassessment is:

> Hard recursive sharing is not the preferred approach for a 10× target. Controlled post-training experiments show that retaining explicit depth-specific low-rank residuals materially improves the quality/capacity frontier. A scale-normalized 2.34375%-rank SoftShare representation reaches perplexity ×1.0776 versus a Q4 teacher in the controlled model, though it is only 9.106× for that complete tiny model. Exact tiny-model 10× points remain substantially worse. For the exact 4,368,438,912-byte Mistral-7B Q4_K_M baseline, the rank-96/KV64 SoftShare design is structurally 11.860× on weights+4K KV, leaving 85,478,016 B of equal-common-scratch headroom before 10×. Real-model quality is not yet measured.

Do **not** state that LARC has demonstrated 10× measured RAM/VRAM or retained real-model intelligence yet.

## Open hard gates

1. **Run the synthetic streaming-converter CI test** and resolve any implementation failures.
2. **L3 real Mistral conversion.** Obtain tensor-range access to the independent pretrained checkpoint and generate a real `.larc` artifact.
3. **Real-model quality.** Same-token perplexity/tasks/generation vs the named Q4_K_M deployment; use validation-gain/byte to allocate rescue ranks.
4. **Complete real file gate.** Final `.larc` must be ≤10% of the exact named baseline unless the claim is explicitly resident-memory-only.
5. **Long-context quality.** Validate the chosen KV rank at realistic context, not just byte arithmetic.
6. **L4 measured memory.** CPU RSS and/or CUDA/Metal peak device memory at the same context, plus throughput.
7. **Competitive 10× alternatives.** Compare SoftShare to a 10× smaller dense/distilled model and other feasible iso-byte representations; do not optimize beyond 10× unless needed for overhead headroom.
