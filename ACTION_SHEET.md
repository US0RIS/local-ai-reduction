# LARC Action Sheet

Canonical technical record for **LARC — Local Adaptive Representation & Compute**. Artifact authority is `benchmarks/INDEX.json`. This sheet is deliberately conservative: synthetic success does not override negative pretrained-model evidence, component ratios are never promoted as whole-model ratios, and measured process memory is kept separate from serialized/model-byte arithmetic.

## Objective

Reduce local-LLM **peak resident inference memory by roughly 10–30× versus a named competitive Q4-class baseline at the same context length**, while retaining useful capability. The representation must eventually support direct compressed-domain CPU/GPU execution rather than reconstructing dense weights.

Evidence levels: **L0** format; **L1** operator/runtime; **L2** controlled model; **L2C** controlled post-training conversion; **L3** independent pretrained LLM; **L4** measured deployment hardware.

## Evidence rules now enforced

1. Current claims must name baseline and context.
2. Quality representation must match byte accounting before promotion.
3. Calibration/training data must be disjoint from final evaluation where applicable.
4. Synthetic results cannot override negative real-model evidence.
5. Operator/component success is not end-to-end model quality.
6. Real-model gates must be committed before observing results.
7. Cross-operator quality gates must aggregate **site-normalized** errors before heterogeneous cross-site aggregation; raw output-energy sums may not be the sole gate.
8. If an FP32/unquantized structural ceiling already fails, later packed failure may not be blamed on quantization.
9. A competitive external baseline is required before any new real-model compression claim.
10. Serialized-file reduction, modeled tensor memory, measured RSS, and VRAM are independent evidence axes.

---

# Run 5 — historical controlled result

Run 5 remains the strongest internally consistent synthetic/control result:

- function-space prefit + hard-projected `Q4_GROUP64` shared weights;
- rank16 Q2/E4M3 latent KV with deterministic Q4 bases and both inverse-Grams;
- five seeds `3,7,11,19,23`;
- context64 mean PPL ratio **1.01208×** the project's simple row-Q4 teacher;
- mean PPL ratio **1.33287×** FP32 teacher;
- modeled direct-packed reduction **11.825×** at context64 and **10.499×** at context8K;
- native CPU L1 primitives separately validate group64-Q4 GEMV and packed Q2/E4M3 attention arithmetic.

This is retained as codec/runtime engineering evidence only. The recurrent sharing and universal low-rank assumptions that produced the ratio did not transfer post hoc to SmolLM2-135M.

---

# Runs 6–8 — real-model falsifications

Model throughout: **`HuggingFaceTB/SmolLM2-135M`**, checkpoint `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

## Run 6 — low rank + direct depth sharing

Activation-aware reduced-rank projection over 49 q/k/v/o/gate/up/down sites:

- rank32 median held-out operator NMSE **0.28662**; only **8.16%** below 0.05;
- rank64 median **0.24243**; only **20.41%** below 0.05.

Direct depth sharing was also severe. A recovered 4→1 share of layers14–17 produced:

- group weight reduction **3.787×**;
- only **1.084×** modeled whole-weight reduction;
- PPL **1.95778×** the internal row-Q4 teacher after recovery.

Decisions: `fail_low_rank_projection`, `fail_current_sharing_recipe`.

## Run 7 — segmented shared output bases

Representation `W_i ≈ B_g C_i` retained layer-specific coefficients but shared output bases.

- strict gate: no group qualified;
- balanced Q/K groups: only **1.02718×** modeled whole-model reduction and PPL **2.25610×** row-Q4;
- aggressive Q/K/O: **1.08716×** reduction and PPL **3.74837×** row-Q4.

Decision: `fail_current_shared_basis_recipe`.

## Run 8 — residual VQ / residual product quantization

Naive activation-RMS-weighted residual vector dictionaries failed catastrophically.

Residual PQ examples:

| nominal payload | modeled whole-weight reduction vs FP16 | NLL | PPL ratio vs FP32 |
|---:|---:|---:|---:|
| 1.0 bpw | **2.409×** | **16.6705** | ~479,749× |
| 1.5 bpw | **1.937×** | **19.8977** | ~12.1M× |
| 2.0 bpw | **1.620×** | **14.6485** | ~63,517× |

Decisions: `fail_naive_additive_vq`, `fail_naive_rpq`.

**Combined Runs 6–8 conclusion:** broad post-hoc low-rank sharing and naive Euclidean/diagonal-RMS vector coding are not credible routes to the required real-model compression.

---

# Run 9 — competitive llama.cpp deployment baseline — active

PR #17. Current workflow has successfully:

- built current llama.cpp unified CPU app;
- converted the exact SmolLM2 checkpoint to F16 GGUF;
- produced `Q4_K_M` and `Q2_K` GGUFs;
- fetched canonical WikiText-2 raw test;
- captured exact provenance;
- **completed full WikiText-2 perplexity measurement**.

It is currently measuring process MaxRSS, then will run PP/TG throughput and assemble the authoritative deployment artifact.

Final Run-9 contract:

- F16 quality reference;
- **Q4_K_M** named competitive Q4 baseline;
- **Q2_K** mature low-bit comparison;
- full WikiText-2 PPL at context512;
- exact GGUF bytes/SHA256;
- GNU `time -v` MaxRSS at context64/2048/8192, mmap and no-mmap, three repetitions;
- allocator/model/KV/compute diagnostics kept separate from process MaxRSS;
- prompt processing and decode throughput;
- exact llama.cpp commit and runner hardware.

No full Run-9 numeric quality/RSS claim is promoted until its final artifact completes.

---

# Run 9A + Run 12 — exact Q4-relative serialized-file feasibility bound

Run 9A reproduced the Run-9 conversion/quantization path but stopped immediately after exact file-size and parameter-count measurement. Its result is committed as `run9a_fast_file_bound_input.json`.

Measured inputs:

- llama.cpp commit: `687e7789271ec1276e3470f158428e11a4f80b6f`;
- parameter count: **134,515,008**;
- F16 GGUF: **270,885,504 B**, effective total-file rate **16.11035 bpw**;
- Q4_K_M GGUF: **105,453,696 B**, effective total-file rate **6.271639 bpw**;
- Q2_K GGUF: **88,201,344 B**, effective total-file rate **5.245591 bpw**.

## Exact target budgets versus measured Q4_K_M

| target | maximum entire candidate file | required effective total-file bpw |
|---:|---:|---:|
| **10×** | **10,545,369.6 B** | **0.6271639 bpw** |
| **20×** | **5,272,684.8 B** | **0.3135819 bpw** |
| **30×** | **3,515,123.2 B** | **0.2090546 bpw** |

This is exact serialized-file arithmetic for the measured Q4_K_M file and llama-bench parameter count.

Impossible-best-case same-parameter zero-metadata bounds:

- 4 bpw → at most **1.568×** vs measured Q4_K_M;
- 2 bpw → **3.136×**;
- 1.58 bpw → **3.969×**;
- **1 bpw → 6.272×**;
- **0.5 bpw → 12.543×**;
- **0.25 bpw → 25.087×**.

## Run-12 decision

A conventional fixed-width **same-parameter ≥1-bit codec cannot mathematically reach even the 10× serialized-file target**, before metadata. The entire 10× file budget is only 0.627 bits per original parameter including container metadata, tokenizer data, scales, codebooks, indices, residuals, and everything else.

Therefore at least one of the following is mandatory for 10–30× vs Q4:

1. substantial parameter elimination/reuse;
2. genuinely sub-bit average structural/entropy coding;
3. a smaller/structured vocabulary interface;
4. a model trained for compressibility rather than converted post hoc;
5. some combination of the above.

The **total resident-memory** portion of Run 12 remains pending measured Run-9 RSS/allocator data. File bpw must not be substituted for RSS or VRAM.

Canonical partial result: `RUN12_FILE_BOUND_STATUS.json`.

---

# Run 10 — maintained optimized W2A16G64 reference — active

PR #18. Reference: Intel AutoRound `W2A16G64`.

Tracks:

- source SmolLM2 under one HF evaluator;
- pure RTN W2 floor;
- tuned AutoRound W2: 200 iterations, 128 calibration samples, sequence length2048, `enable_alg_ext=true`.

The evaluator uses identical token IDs and context512 boundaries for source/RTN/tuned models and batches equal-length windows only for throughput. Full WikiText-2 source evaluation is currently running.

This external PTQ arm matters because Run 11 shows that second-order/rotation-aware W2 is materially better than naive Q2. If competent optimized W2 is still far from useful quality, further PTQ-only work cannot solve the exact <0.627-bpw target by itself anyway.

---

# Run 11 — positive second-order + structured-rotation mechanism

49 real q/k/v/o/gate/up/down sites, calibrated on WikiText-2 train and evaluated on disjoint WikiText-2 test activations.

Diagnostic Q2 representation: four asymmetric levels, group64, FP16 min+scale metadata, approximately 2.5 bpw / 6.4× matrix reduction versus FP16.

| representation | median held-out output NMSE | fraction <0.05 | reduction vs FP16 matrix |
|---|---:|---:|---:|
| plain Q2 | **0.16072** | 6.12% | **6.400×** |
| block-GPTQ-style Q2 | 0.16527 | 6.12% | 6.400× |
| Hadamard Q2 | 0.15555 | 6.12% | 6.396× |
| **Hadamard + block-GPTQ Q2** | **0.09555** | **36.73%** | **6.396×** |

Best-mechanism operator medians:

- Q **0.03909**;
- K **0.02994**;
- V **0.19110**;
- O **0.14008**;
- gate **0.08547**;
- up **0.15635**;
- down **0.15705**.

The combination cuts median operator NMSE by **40.55%** versus plain Q2 at essentially unchanged bytes. Simple 1–5% FP16 outlier-column escape and dense learned rotations did not improve the best error/byte point.

**Decision:** retain Hadamard + covariance-aware/second-order rounding as a component mechanism, especially for Q/K. Do not promote whole-model W2.

---

# Run 13 — full-rank shared base + low-rank layer residuals

Hypothesis: `W_layer ≈ B_group + U_layer V_layer^T`.

All 210 main decoder projection matrices were tested with 2/3/6 physical bases and rank8/16/32 layer residuals, using activation-aware fitting and representation-matched Q4_GROUP64 bases/factors.

Byte economics were structurally relevant:

- 2 bases/rank8: **10.583×** main-projection reduction;
- 2/r16: **8.561×**;
- 2/r32: **6.194×**;
- 3/r32: **5.134×**;
- 6/r32: **3.392×**.

But normalized quality failed:

- 2/r8 median site NMSE **0.5042**;
- 3/r32 **0.3783**;
- 6/r32 **0.3402**;
- only ~3–4% of sites below 0.05.

At 3 bases/rank32, Q/K were best but still poor (Q 0.1291, K 0.1085); V/O/MLP were 0.30–0.65 median NMSE.

Most importantly, the unquantized FP32 residual-factor ceilings were nearly as bad, proving Q4 factor quantization was not the main failure.

The original precommitted raw-energy global NMSE gate formally reported passes; that metric was audited as invalid because heterogeneous output-energy weighting masked severe normalized per-site failures. The formal result remains preserved for auditability, but substantive decision is:

`fail_current_additive_residual_sharing_ranks_le_32`.

---

# Run 14 — trained nonlinear shared MLP + depth FiLM

Run 14 asked whether local function training could rescue the dominant MLP byte pool after linear residual sharing failed.

One physical full-rank nonlinear MLP was shared across either 5 or 10 logical layers. The shared gate/up/down matrices and tiny layer-specific FiLM scales were trained directly against real teacher MLP input/output pairs. The final packed representation used Q4_GROUP64 shared matrices + FP16 FiLM.

## 5 logical layers / physical MLP

- MLP-pool byte reduction: **4.906×**;
- untrained mean-MLP median NMSE: **0.9830**;
- trained FP32 median: **0.9248**;
- packed median: **0.9253**;
- packed p90: **0.9599**;
- layers below 0.10 NMSE: **0/30**.

## 10 logical layers / physical MLP

- MLP-pool byte reduction: **9.633×**;
- untrained median: **0.9968**;
- trained FP32 median: **0.9419**;
- packed median: **0.9419**;
- packed p90: **0.9826**;
- layers below 0.10: **0/30**.

**Decision:** `fail_posthoc_local_shared_mlp_function_distillation`.

The near identity of FP32 and packed errors again proves quantization is not the bottleneck. This rejects the tested **local post-hoc** nonlinear sharing recipe; it does not reject architectures globally uptrained or pretrained with recurrence/sharing.

Canonical result: `RUN14_FINAL_STATUS.json`.

---

# Run 15 — tied embedding/head low-rank factorization — provisional failure, validation pending

SmolLM2's tied 49,152×576 vocabulary matrix has **28,311,552 parameters**, about **21.05%** of the model. If it remains unchanged, even making every other parameter free caps weight reduction at only **4.751×**.

Run 15 factorizes the tied matrix as `E ≈ A B`, shares A/B between input lookup and output head, and evaluates both FP32 factors and Q4_GROUP64 factors on an 8,192-prediction WikiText-2 test slice.

Initial measured result is extremely poor:

- rank64: **8.896×** tied-matrix reduction; packed integrated PPL ratio **1,387×** reference;
- rank128: **4.448×**; packed integrated PPL ratio **3,727×**;
- rank192: **2.965×**; packed integrated PPL ratio **8,378×**;
- rank256: **2.224×**; packed integrated PPL ratio **9,960×**.

Even FP32 factor ceilings and head-only replacement are catastrophic, suggesting the language-critical tied vocabulary geometry is not captured by a low global rank.

However, because integrated PPL worsens non-monotonically as rank rises, **Run 15 is not yet accepted as architectural evidence**. Run 15B is active as a hard implementation control:

- ranks384/448/512/576;
- rank576 FP32 must reconstruct embedding NMSE ≤1e-8;
- rank576 head PPL ratio must be within 1e-4 of 1.0;
- integrated PPL ratio within 1e-3 of 1.0.

If the full-rank control fails, Run 15 is invalid and the harness must be fixed. If it passes, the low-rank failure becomes credible.

---

# Run 16 — global recursive decoder distillation pilot — queued

This is the first project experiment that imposes extreme projection sharing and then globally trains the **entire language model** under teacher distillation.

Representation for q/k/v/o/gate/up/down:

`W_layer = B_(layer mod P) + U_layer V_layer`

with average shared-base initialization and randomized-SVD rank8 residual initialization.

Two arms:

- P=2 physical phases/rank8 → expected **11.152×** main-projection parameter reduction;
- P=3/rank8 → **8.130×**.

Original tied embedding/head remains unchanged and frozen to isolate decoder sharing. Layer-specific RMSNorm remains trainable.

Global budget:

- 160 steps × seq64 = ≤10,240 sampled train tokens;
- WikiText-2 raw train;
- `0.75 × forward-KL(T=2) + 0.25 × next-token CE`;
- held-out PPL on 4,096 WikiText-2 test predictions.

Frozen pilot gate:

- pass: ≥8× main-projection parameter reduction and PPL ≤1.5× teacher;
- promising/extend: ≥8×, PPL ≤3×, and ≥40% best training-loss improvement.

A failure rejects only this short-budget conversion recipe. It must not be generalized to literature-scale recursive uptraining. The relevant external reference, **Relaxed Recursive Transformers (ICLR 2025 / arXiv:2410.20672)**, uses moderate sharing, depth-wise LoRA, careful initialization, knowledge distillation, and billions to tens of billions of uptraining tokens; those economics/training budgets differ radically from LARC's 10×-class target.

---

# Run 17 — direct-packed tied-vocabulary product quantization — queued

Run 17 attacks the vocabulary floor without assuming low global rank.

Representation for token `t`:

`E[t] ≈ norm[t] × concat(C_s[code[t,s]])`.

Storage:

- one FP16 norm/token;
- one uint8 code/token/subspace;
- 256 FP16 centroids/subspace;
- no dense vocabulary shadow.

Input inference is direct centroid gathering. Output logits are also direct: compute 256 centroid dot products per subspace, then gather/sum the selected entries for each token and multiply by token norm. A semantic test compares this packed-head computation against a separately decoded dense reference.

Tested subspace dimensions 8/12/16/24/32 span exactly:

| subdim | tied-vocab bytes | reduction vs tied Q4_GROUP64 | effective embedding bpw |
|---:|---:|---:|---:|
| 8 | 3,932,160 | 3.825× | 1.1111 |
| 12 | 2,752,512 | 5.464× | 0.7778 |
| 16 | 2,162,688 | 6.955× | 0.6111 |
| 24 | 1,572,864 | 9.563× | 0.4444 |
| 32 | 1,277,952 | 11.769× | 0.3611 |

Frozen pass gate:

- ≥5× tied-Q4 reduction;
- occurrence-weighted embedding NMSE ≤0.05;
- head-only PPL ≤1.05×;
- integrated input+head PPL ≤1.10×;
- packed semantic error ≤1e-4.

Borderline: ≥4×, NMSE ≤0.10, head-only ≤1.15×, integrated ≤1.50×.

The five frozen subdims are now scheduled as **independent matrix jobs** to avoid a serial CPU timeout; this changes execution scheduling only, not the experiment.

---

# Run 18 — combined description-budget envelope

Run 18 does not test intelligence. It combines the exact byte contracts already frozen for Runs 16 and 17 and asks whether those components, **if they pass quality**, are sufficient for the measured 10× serialized-file target.

Measured 10× budget: **10,545,369.6 B**.

## Conservative overhead allowance

Measured F16 GGUF is 270,885,504 B. Exact unique FP16 tensors are:

`134,515,008 × 2 = 269,030,016 B`.

Run 18 reserves the entire difference, **1,855,488 B**, as a conservative allowance for tokenizer/container/tensor metadata/alignment. No reduction in this overhead is assumed.

All 35,136 remaining unique non-embedding/non-main-projection parameters are also kept at FP16: **70,272 B**.

## Recursive decoder Q4_GROUP64 bytes

P=2/rank8:

- shared physical bases **3,760,128 B**;
- depth LoRA **1,569,600 B**;
- total **5,329,728 B**.

P=3/rank8:

- shared bases **5,640,192 B**;
- depth LoRA **1,569,600 B**;
- total **7,209,792 B**.

## Combined conservative totals

| physical phases | vocab PQ subdim | conservative total | modeled reduction vs measured Q4_K_M | 10× headroom |
|---:|---:|---:|---:|---:|
| 2 | 8 | 11,187,648 B | 9.426× | −642,278 B |
| **2** | **12** | **10,008,000 B** | **10.537×** | **+537,370 B** |
| 2 | 16 | 9,418,176 B | 11.197× | +1,127,194 B |
| 2 | 24 | 8,828,352 B | 11.945× | +1,717,018 B |
| 2 | 32 | 8,533,440 B | 12.358× | +2,011,930 B |
| 3 | 8 | 13,067,712 B | 8.070× | −2,522,342 B |
| 3 | 12 | 11,888,064 B | 8.871× | −1,342,694 B |
| 3 | 16 | 11,298,240 B | 9.334× | −752,870 B |
| 3 | 24 | 10,708,416 B | 9.848× | −163,046 B |
| **3** | **32** | **10,413,504 B** | **10.127×** | **+131,866 B** |

## Run-18 decision

**The current structural byte contracts are sufficient in principle for the first 10× serialized-size milestone without requiring sub-1-bit physical decoder weights, if Run 16 and Run 17 survive quality.**

- P=2/rank8 requires PQ subdim12 or more aggressive under this allowance.
- P=3/rank8 crosses 10× only with subdim32.

This is an important refinement of the Run-12 conclusion. Run 12 proves same-parameter ≥1-bit quantization is insufficient. Run 18 shows that after substantial parameter sharing and vocabulary composition, **ordinary Q4_GROUP64 physical decoder matrices can fit the first 10× description envelope**. Ternary/sub-1-bit training remains useful for more headroom, resident-memory reduction, and 20–30×, but is not inherently required for the initial file-size milestone.

Canonical arithmetic: `RUN18_DESCRIPTION_BUDGET.json`; generator `tools/run18_description_budget.py`; detailed protocol `docs/RUN18_DESCRIPTION_BUDGET.md`.

No quality, native runtime, RSS, VRAM, TTFT, or throughput claim is made by Run 18.

---

# Current architecture conclusion

The exact Run-12 rate bound changes how the project should be framed:

> **This is no longer primarily a quantization problem. It is a model-description-length problem.**

A 10× candidate must fit **everything** in ~10.55 MB, equivalent to only 0.627 total bits per original SmolLM2 parameter. Conventional same-parameter 1–2 bit quantization cannot reach that. The model must have far fewer independently described degrees of freedom.

Run 18 further shows that the **byte economics are no longer the blocker** for one concrete composite architecture: if the P=2/rank8 recursive decoder and a Run-17 PQ point at subdim12 or higher both preserve quality, the combined Q4 physical-weight description fits the measured 10× file budget even after a conservative 1.855 MB overhead allowance.

Real-model evidence now says:

### Falsified broad post-hoc routes

1. universal low-rank per-matrix projection;
2. direct whole-block depth sharing;
3. shared low-rank output bases as the main codec;
4. naive residual VQ/RPQ;
5. full-rank shared bases + rank≤32 local linear residuals at 5–15-layer sharing spans;
6. local nonlinear MLP sharing with only small FiLM depth state.

### Positive reusable mechanisms

1. native `Q4_GROUP64` GEMV arithmetic;
2. native Q2/E4M3 packed latent-attention arithmetic;
3. Q/K consistently tolerate more aggressive representation than V/O/MLP;
4. Hadamard rotation + covariance-aware/second-order rounding materially improves W2 operator fidelity;
5. structural sharing can easily achieve 5–11× **byte/parameter economics**, but post-hoc local fitting has not preserved function;
6. the tied vocabulary matrix is a mandatory compression target;
7. exact description-budget arithmetic now identifies P=2/r8 + vocabulary PQ subdim≥12 as a 10×-capable serialized contract **if quality survives**;
8. rigorous provenance/gating prevents synthetic/component arithmetic from becoming false whole-model claims.

### Primary remaining hypothesis

The model must likely be **trained or globally distilled into a low-description-length architecture**:

- a small number of physical recurrent/recursive decoder templates;
- inexpensive depth conditioning or low-rank relaxation;
- low-bit physical weights learned during optimization, potentially ternary/BitNet-like rather than PTQ-only;
- a structured/compositional vocabulary interface rather than an unchanged 49k×576 dense matrix;
- packed latent KV and direct compressed-domain kernels.

Native ternary training is relevant because BitNet b1.58 demonstrates that very low-bit weights can remain competitive when low precision is built into training, but Run-12 arithmetic also shows **1.58 bits alone is nowhere near enough**: a same-parameter 1.58-bpw payload can only be ~3.97× smaller than the measured Q4_K_M file before overhead. Parameter reduction and low-bit training must be combined for the 20–30× extension.

---

# Current claim boundary

> **No usable real-pretrained LARC candidate exists yet.**

What is now established:

- exact measured Q4_K_M file-rate target arithmetic;
- several real-model structural mechanisms have been falsified post hoc;
- one real-model low-bit mechanism (Hadamard + second-order rounding) is positive at operator level;
- local nonlinear sharing fails even before quantization;
- vocabulary compression is mathematically mandatory;
- a concrete Run-16/17 composite has sufficient modeled **serialized-byte economics** for 10× under conservative overhead, but its quality is unproven;
- global recursive distillation and direct-packed vocabulary PQ are the next active quality tests.

Do **not** claim:

- 10× real-model compression achieved;
- Q4_K_M quality parity achieved;
- 10× RSS or VRAM achieved;
- Run-18 modeled 10× byte feasibility as an executed `.larc` file;
- that Run-11's ~6.4× matrix-vs-FP16 ratio is a Q4-relative whole-model ratio;
- that Run-13/14 component byte ratios translate into a usable model;
- that the provisional Run-15 low-rank failure is final before the rank576 sanity control;
- native end-to-end LARC inference.

---

# Highest-priority work

1. **Finish full Run 9** and commit actual Q4_K_M/Q2_K PPL, MaxRSS, allocator pools, and throughput.
2. **Complete Run 12 total-memory bound** using measured Run-9 RSS, distinguishing removable weight residency from KV/runtime fixed floors.
3. **Validate Run 15 with rank576** before accepting its severe low-rank conclusion.
4. **Execute Run 17** direct-packed vocabulary PQ. The first strategically important quality point is subdim12 because Run 18 shows it is the least aggressive P=2 vocabulary setting that still closes the 10× file budget.
5. **Execute Run 16** global recursive rank8 distillation. P=2 has the best 10× description headroom; P=3 is the less aggressive structural-quality control but only crosses 10× with the most aggressive tested vocabulary PQ.
6. **Finish Run 10** optimized W2. If W2 remains weak, PTQ-only work becomes secondary because the exact 0.627-bpw target already requires structural reduction regardless.
7. If Run 16 short-budget recovery is promising, extend uptraining before changing representation, then quantize/QAT the physical bases and adapters.
8. If Run 17 passes, implement native PQ embedding/output-head kernels and full-corpus validation.
9. If Run 16 is not promising and Run 17 also fails, move to a **jointly trained recurrent student with a redesigned/factorized/compositional tokenizer interface** rather than further post-hoc conversion.
10. Once a real L3 representation survives full-corpus quality, integrate it with the existing packed Q2/E4M3 KV primitive and measure real RSS/TTFT/tokens/s under the same protocol as Q4_K_M.
11. Only then pursue CUDA/Metal L4 and the 20–30× extension.

The **20–30× objective remains open**. The exact file-rate evidence now makes clear that achieving it requires a model whose learned information content is fundamentally smaller than the original independent parameter tensor set—not merely a more aggressive conventional quantizer.
