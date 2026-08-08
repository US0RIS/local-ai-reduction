# LARC Action Sheet

Canonical technical record for **LARC — Local Adaptive Representation & Compute**. Artifact authority is `benchmarks/INDEX.json`. The current real-model summaries are `RUN6_FINAL_STATUS.json`, `RUN7_FINAL_STATUS.json`, and `RUN8_FINAL_STATUS.json`; Run 5 is now historical controlled-model evidence rather than the preferred real-model architecture.

## Objective

Reduce local-LLM **peak resident inference memory by roughly 10–30× versus a named competitive Q4-class baseline at the same context length**, while retaining useful capability. Every promoted claim must name baseline, context, byte pools, quality representation, calibration/evaluation separation, and whether execution/memory is measured or modeled.

Evidence levels: **L0** format, **L1** operator/runtime, **L2** controlled model, **L2C** controlled post-training conversion, **L3** independent pretrained LLM, **L4** measured deployment hardware.

---

# Historical controlled result retained — Run 5

Run 5 established that the project machinery can produce an internally consistent aggressive representation on a synthetic character LM:

- teacher-layer function prefit + hard-projected `Q4_GROUP64` shared weights;
- rank16 Q2/E4M3 latent KV with deterministic Q4 bases and both inverse-Grams;
- five seeds `3,7,11,19,23`;
- mean PPL ratio **1.01208×** the project's simple row-Q4 teacher at context64;
- mean PPL ratio **1.33287×** FP32 teacher;
- modeled packed reduction **11.825×** at context64 and **10.499×** at context8K;
- separate native CPU L1 primitives for group64-Q4 GEMV and packed Q2/E4M3 latent attention.

This result remains useful for codec/runtime engineering, but **its architectural transfer assumptions are not promoted to real pretrained models** after Runs 6–8.

Key artifacts: `run5_e4m3_multiseed.json`, `run5_packed_context_sweep.json`, `run5_native_q4_group64.json`, `run4_native_q2_attention.json`.

---

# Run 6 — first real pretrained falsification

Model: **`HuggingFaceTB/SmolLM2-135M`**, checkpoint `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

Run 6 tested the two assumptions carrying most of the Run-5 compression ratio: broad low-dimensional operator geometry and aggressive cross-depth block sharing.

## Activation-aware reduced-rank result

49 real projection sites were measured across layers 0/5/10/15/20/25/29 and q/k/v/o/gate/up/down.

- rank32 median held-out operator NMSE: **0.28662**;
- rank32 fraction below 0.05 NMSE: **8.16%**;
- rank64 median: **0.24243**;
- rank64 fraction below 0.05: **20.41%**.

Precommitted gate: **`fail_low_rank_projection`**.

The failure is not uniform by operator. At rank128 the strongest families are K and Q:

- K: all sampled layers below 0.05 NMSE;
- Q: 4/7 sampled layers below 0.05;
- V, O, MLP up/gate, and especially down are materially worse.

Conclusion: low rank is an **operator-specific local property**, not a universal model representation.

## Raw layer interchangeability

Single neighboring-layer replacement had median PPL ratio **1.261×**. Contiguous exact sharing was severe:

- layers 8–9: **29.56× PPL**;
- layers 14–17: **5.42×**;
- layers 22–25: **5.82×**.

## Partial 4→1 recovered conversion

Layers 14–17 were collapsed into one physical block, initialized from their FP32 parameter mean, hard-projected to group64 Q4, and recovered for 24 QAT/distillation steps against a row-Q4 teacher.

- group weight reduction: **3.787×**;
- whole-model modeled weight reduction from this one group: **1.084×**;
- row-Q4 teacher NLL: **4.96356**;
- recovered shared NLL: **5.63537**;
- PPL ratio vs row-Q4: **1.95778×**.

Precommitted gate: **`fail_current_sharing_recipe`**.

**Run-6 conclusion:** direct post-hoc whole-block sharing and universal low-rank activation projection are falsified on SmolLM2-135M.

Artifacts: `run6_real_model_falsification.json`, `run6_activation_aware_projection.json`, `run6_partial_real_conversion.json`, `RUN6_GATE.json`, `RUN6_FINAL_STATUS.json`.

---

# Run 7 — segmented shared-basis follow-up

Run 7 preserved layer-specific functions using like-operator shared output bases:

`W_i ≈ B_g C_i`

with one basis `B_g` per operator/depth group and unique coefficients `C_i` per logical layer. FP32 and row-Q4 source representations were fitted independently; deployment ranks were chosen only from held-out **post-Q4_GROUP64-factor** operator error.

Baseline on the fixed custom evaluation slice:

- FP32 NLL: **3.88795**;
- simple row-Q4 NLL: **5.09597**;
- row-Q4 PPL ratio vs FP32: **3.34683×**.

This establishes that the project's row-Q4 is a weak research reference, not a competitive deployment baseline.

Results:

- strict gate: no group qualifies;
- balanced: six Q/K groups qualify, but whole-model modeled reduction is only **1.02718×** and Q4-factor PPL is **2.25610×** row-Q4;
- aggressive: eight Q/K/O groups, **1.08716×** modeled reduction, PPL **3.74837×** row-Q4.

Precommitted gate: **`fail_current_shared_basis_recipe`**.

**Run-7 conclusion:** Q/K contain useful shared low-rank structure, but they are too small a byte pool to drive extreme whole-model compression. The dominant MLP matrices do not tolerate this shared-basis recipe at useful ranks.

Artifacts: `run7_shared_basis_real_model.json`, `RUN7_GATE.json`, `RUN7_FINAL_STATUS.json`.

---

# Run 8 — aggressive layer-preserving vector quantization

Run 8 deliberately abandoned low-rank sharing and targeted the dominant projection byte pool while keeping every logical matrix distinct.

## Run 8A: residual VQ

Representation:

- 32-weight vectors;
- input-RMS activation weighting as a diagonal-Hessian proxy;
- FP16 per-vector magnitude;
- 16-entry FP16 residual codebooks shared by like operator / 10-layer group;
- 4-bit index per residual stage.

At the most aggressive tested point, nominal target payload was **0.75 bpw** and modeled whole-model weight reduction was about **2.75×**, but NLL rose from FP32 ~**3.589** to ~**18.07**. Even the richest tested representation remained unusable.

Calibration residual energy decreased monotonically, ruling out a trivial sign/scale reconstruction bug. The 32-D 16-way dictionary simply leaves too much geometry unexplained.

Gate: **`fail_naive_additive_vq`**.

## Run 8B: residual product quantization

The 32-D direction was split into four independently coded 8-D subspaces; codebooks became per-layer/per-operator. This removes depth sharing and greatly increases codeword combinatorics.

Measured points:

| nominal target payload | modeled whole-weight reduction | NLL | PPL ratio vs FP32 |
|---:|---:|---:|---:|
| 1.0 bpw | **2.409×** | **16.6705** | ~479,749× |
| 1.5 bpw | **1.937×** | **19.8977** | ~12.1M× |
| 2.0 bpw | **1.620×** | **14.6485** | ~63,517× |

The richest RPQ stage reduces sampled activation-weighted vector residual energy to roughly 20% for much of the model, but that is still far too much perturbation for end-to-end inference.

Gate: **`fail_naive_rpq`**.

**Run-8 conclusion:** naive Euclidean/diagonal-RMS vector dictionaries are not a credible primary extreme-weight codec for this model. More centroids or greedy residual stages are not the next priority.

Artifacts: `run8_additive_vq_real_model.json`, `RUN8_GATE.json`, `run8_residual_pq_real_model.json`, `RUN8_RPQ_GATE.json`, `RUN8_FINAL_STATUS.json`.

---

# Current real-model conclusion after Runs 6–8

The following mechanisms are now **falsified as broad post-training solutions on SmolLM2-135M** under the tested protocols:

1. universal low-rank activation projection;
2. direct whole-block cross-depth sharing;
3. segmented shared output bases as the main weight codec;
4. naive sub-2-bit residual vector quantization;
5. naive residual product quantization.

These failures materially change the project direction. Do **not** continue optimizing the Run-5 recurrent/shared architecture as though real-model transfer were merely unfinished.

The strongest positive reusable components are narrower:

- native group64-Q4 weight GEMV;
- native packed Q2/E4M3 latent attention arithmetic;
- evidence that Q/K activations can have substantially lower effective rank than MLP projections;
- the experiment/provenance infrastructure for representation-matched real-model testing.

---

# Competitive-baseline gap

The project still lacks a committed measured optimized deployment baseline. This is now blocking new compression claims because the simple row-Q4 reference performs very poorly on the real-model slice.

The next external baseline must include actual llama.cpp **Q4_K_M** and **Q2_K** for SmolLM2-135M, with:

- exact GGUF file bytes and hashes;
- WikiText-2 perplexity;
- process MaxRSS at context64/2K/8K;
- mmap and non-mmap/load-mode measurements;
- prompt-processing and token-generation throughput;
- exact llama.cpp commit/version and runner hardware class.

A separate maintained **W2A16G64 optimized quantization** reference should establish how much quality a competent 2-bit optimizer can retain before LARC attempts another custom low-bit representation.

---

# Current claim boundary

> **No usable real-pretrained LARC candidate exists yet.** The ~10.5–11.8× result remains controlled/synthetic evidence only. Runs 6–8 show that the architectural mechanisms producing most of that ratio do not transfer directly to SmolLM2-135M under the tested post-training protocols.

Do not claim:

- 10× real-model compression;
- Q4_K_M parity;
- measured LARC RSS/VRAM savings;
- standard-benchmark quality for Runs 6–8;
- native end-to-end LARC inference;
- that naive low-rank/shared/VQ mechanisms remain the preferred architecture.

---

# Highest-priority work now

1. **Competitive deployment baseline:** measure SmolLM2 Q4_K_M and Q2_K in current llama.cpp at 64/2K/8K context, including WikiText-2 PPL, MaxRSS, and throughput.
2. **Strong W2 reference:** tuned AutoRound/GPTQ-class 2-bit group64 result on the same pretrained checkpoint. The project must understand the real PTQ frontier before designing another sub-2-bit scheme.
3. **Second-order / rotation / outlier-aware weight compression:** only after 1–2 establish the target. Candidate mechanisms include block-Hessian/activation-covariance transforms, Hadamard/QuaRot-style rotations, optimized discrete rounding, outlier escape/residual channels, and learned codebook/index refinement.
4. **Distilled/learned architecture path:** if competent W2 remains far from useful quality, treat >10× vs Q4 as a training/distillation problem rather than a post-training codec problem. Direct post-hoc sharing failed; a model trained from the outset for recurrent/shared/dictionary structure is a distinct hypothesis and remains open.
5. **Runtime integration remains required:** once a real representation passes quality, integrate its packed weight primitive with the existing packed Q2/E4M3 KV path and measure actual RSS/TTFT/tokens/s under the same protocol as Q4_K_M.
6. **Standard evaluation:** move real candidates from the short custom slice to WikiText-2 and task/generation tests before any L3 promotion.
7. **L4 hardware:** only after a real candidate survives L3 should CUDA/Metal/consumer-CPU measurements drive final claims.

The 20–30× objective remains open. It should not be pursued by extrapolating synthetic compression ratios; it must be rebuilt from real-model evidence.