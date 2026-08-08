# LARC Action Sheet

Canonical technical record for **LARC — Local Adaptive Representation & Compute**. Artifact authority is `benchmarks/INDEX.json`. Current real-model summaries include `RUN6_FINAL_STATUS.json`, `RUN7_FINAL_STATUS.json`, `RUN8_FINAL_STATUS.json`, `RUN11_FINAL_STATUS.json`, and `RUN13_FINAL_STATUS.json`. Run 5 remains historical controlled-model evidence rather than a transferable real-model architecture.

## Objective

Reduce local-LLM **peak resident inference memory by roughly 10–30× versus a named competitive Q4-class baseline at the same context length**, while retaining useful capability. Every promoted claim must name the baseline, context, byte pools, quality representation, calibration/evaluation separation, and whether execution/memory is measured or modeled.

Evidence levels: **L0** format, **L1** operator/runtime, **L2** controlled model, **L2C** controlled post-training conversion, **L3** independent pretrained LLM, **L4** measured deployment hardware.

Evidence rules now include: synthetic success cannot override negative real-model evidence; operator-level success is not end-to-end model quality; quality representation must match byte accounting; calibration/evaluation must be disjoint; and cross-operator gates must not hide normalized site errors behind raw output-energy weighting.

---

# Run 5 — historical controlled result

Synthetic character-LM result:

- teacher-layer function prefit + hard-projected `Q4_GROUP64` shared weights;
- rank16 Q2/E4M3 latent KV with deterministic Q4 bases and K/V inverse-Grams;
- five seeds `3,7,11,19,23`;
- mean PPL ratio **1.01208×** internal row-Q4 at context64;
- mean PPL ratio **1.33287×** FP32 teacher;
- modeled packed reduction **11.825×** at context64 and **10.499×** at context8K;
- native CPU L1 group64-Q4 GEMV and packed Q2/E4M3 latent-attention primitives exist separately.

This remains useful codec/runtime engineering evidence. Its universal low-rank and recurrent-sharing assumptions are **not** promoted to pretrained LLMs.

Key artifacts: `run5_e4m3_multiseed.json`, `run5_packed_context_sweep.json`, `run5_native_q4_group64.json`, `run4_native_q2_attention.json`.

---

# Run 6 — real pretrained falsification

Model: `HuggingFaceTB/SmolLM2-135M`, checkpoint `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

Activation-aware reduced-rank tests across 49 real q/k/v/o/gate/up/down sites:

- rank32 median held-out operator NMSE **0.28662**; only **8.16%** below 0.05;
- rank64 median **0.24243**; only **20.41%** below 0.05.

Gate: `fail_low_rank_projection`.

Cross-depth sharing:

- neighboring single-layer replacement median PPL ratio **1.261×**;
- exact share layers8–9 **29.56× PPL**;
- exact share layers14–17 **5.42×**;
- exact share layers22–25 **5.82×**.

Recovered 4→1 layers14–17:

- group weight reduction **3.787×**;
- whole-model modeled weight reduction **1.084×**;
- row-Q4 teacher NLL **4.96356**;
- recovered shared NLL **5.63537**;
- PPL **1.95778×** row-Q4.

Gate: `fail_current_sharing_recipe`.

**Conclusion:** universal low-rank activation projection and direct post-hoc whole-block sharing are falsified for this real model under the tested protocol.

---

# Run 7 — segmented shared output bases

Representation `W_i ≈ B_g C_i` retained layer-specific coefficients around like-operator/depth shared bases.

The internal row-Q4 baseline was itself weak: FP32 NLL **3.88795**, row-Q4 **5.09597**, PPL ratio **3.34683×** FP32.

Results:

- strict: no group qualified;
- balanced: six Q/K groups, only **1.02718×** whole-model modeled reduction, PPL **2.25610×** row-Q4;
- aggressive: eight Q/K/O groups, **1.08716×** modeled reduction, PPL **3.74837×** row-Q4.

Gate: `fail_current_shared_basis_recipe`.

**Conclusion:** Q/K show useful shared structure, but are too small a byte pool; dominant MLP matrices do not tolerate this representation at useful ranks.

---

# Run 8 — layer-preserving vector quantization

## 8A additive residual VQ

32-weight vectors, activation-RMS weighting, FP16 magnitude, shared 16-entry residual dictionaries, 4-bit indices.

At nominal **0.75 bpw**, modeled whole-weight reduction was ~**2.75×** vs FP16 but NLL rose from ~**3.589** to ~**18.07**. Residual energy decreased monotonically, so this was not a trivial reconstruction bug.

Gate: `fail_naive_additive_vq`.

## 8B residual product quantization

Four 8-D subspaces with per-layer/per-operator codebooks:

| target | modeled whole-weight reduction | NLL | PPL ratio vs FP32 |
|---:|---:|---:|---:|
| 1.0 bpw | **2.409×** | **16.6705** | ~479,749× |
| 1.5 bpw | **1.937×** | **19.8977** | ~12.1M× |
| 2.0 bpw | **1.620×** | **14.6485** | ~63,517× |

Gate: `fail_naive_rpq`.

**Conclusion:** naive Euclidean/diagonal-RMS residual dictionaries are not a credible extreme-weight codec.

---

# Run 9 — competitive llama.cpp deployment baseline — in progress

PR #17. This is the blocking external deployment baseline:

- F16 GGUF quality reference;
- **Q4_K_M** named competitive Q4 baseline;
- **Q2_K** mature low-bit comparison;
- complete WikiText-2 `llama perplexity` at context512;
- exact GGUF bytes/SHA256;
- GNU `time -v` process MaxRSS at context64/2048/8192, mmap and no-mmap, three reps;
- llama.cpp-reported allocator pools kept distinct from process MaxRSS;
- `llama bench` PP at 64/2048/8192 and tg128 after depth64/2048/8064;
- exact llama.cpp commit and CPU-runner provenance.

Two infrastructure-only failures were corrected: current llama.cpp uses the unified `llama` app, and that app requires its CLI/server implementation libraries at link time. The current run has successfully built llama.cpp, converted/quantized SmolLM2, fetched the corpus, and captured provenance; full PPL measurement is in progress.

No Run-9 numeric baseline is promoted until the artifact completes.

---

# Run 10 — maintained optimized W2A16G64 reference — in progress

PR #18. Reference: Intel AutoRound `W2A16G64`.

Tracks:

- source SmolLM2 under one HF evaluator;
- pure RTN W2 floor;
- tuned W2: 200 iterations, 128 calibration samples, sequence length2048, `enable_alg_ext=true`.

All three use identical tokenizer/corpus/context boundaries on the full WikiText-2 test stream. The evaluator was changed from serial windows to batches of four **without changing token IDs or 512-token boundaries**, and asserts that predicted-token count equals corpus tokens minus one.

If tuned W2 enters a useful regime, escalate to the more expensive AutoRoundBest/maximum-quality path. If competent W2 remains poor, the 10–30× objective is increasingly a trained-structure problem rather than another post-training rounding problem.

No Run-10 numeric result is promoted until the artifact completes.

---

# Run 11 — second-order + structured rotation W2 diagnostic

49 real sites at layers0/5/10/15/20/25/29, calibrated on WikiText-2 train and evaluated on disjoint test activations.

Diagnostic Q2 contract: asymmetric four-level group64 + FP16 min/scale, approximately **2.5 bpw / 6.4× matrix reduction vs FP16**.

Aggregate:

| representation | median held-out output NMSE | fraction <0.05 | matrix reduction vs FP16 |
|---|---:|---:|---:|
| plain Q2 | **0.16072** | 6.12% | 6.400× |
| block-GPTQ-style Q2 | 0.16527 | 6.12% | 6.400× |
| Hadamard Q2 | 0.15555 | 6.12% | 6.396× |
| **Hadamard + block-GPTQ Q2** | **0.09555** | **36.73%** | **6.396×** |
| dense learned orthogonal ceiling | 0.15907 | 6.12% | 3.740× |

Hadamard + second-order error feedback improves median NMSE **40.55%** at essentially unchanged bytes.

Operator medians for the best mechanism:

- Q **0.03909**;
- K **0.02994**;
- V **0.19110**;
- O **0.14008**;
- gate **0.08547**;
- up **0.15635**;
- down **0.15705**.

Simple 1–5% FP16 outlier-column escape worsened the best aggregate error/byte point. Dense learned covariance rotations also did not justify transform bytes.

**Decision:** promote Hadamard + block second-order rounding as a **component mechanism only**. Do not promote all-W2, simple FP16 escape, or dense learned rotations.

Canonical artifact: `RUN11_FINAL_STATUS.json`.

---

# Run 12 — exact Q4-relative feasibility bound — calculator committed, awaiting Run 9

The calculator was merged to `main` **before Run-9 results were observed**.

Once Run 9 completes it will calculate, from exact Q4_K_M bytes and llama-bench parameter count:

- Q4 effective total-file bits/original-parameter;
- exact 10×/20×/30× candidate file budgets;
- required effective total-file bpw at each target;
- impossible-best-case zero-metadata same-parameter bounds for 4/3/2.5/2/1.58/1/0.5/0.25 bpw;
- an explicitly illustrative total-RSS floor model, kept separate from measured allocator/RSS evidence.

Known structural arithmetic independent of Run 9:

- Q+K are only **12.5%** of SmolLM2's seven main decoder projection parameters;
- gate/up/down are **75%**.

Therefore aggressive Q/K W2 cannot by itself explain a 10× whole-model/Q4 result.

Generator: `tools/run12_feasibility_bound.py`.

---

# Run 13 — full-rank shared base + low-rank layer residuals

Hypothesis:

`W_layer ≈ B_group + U_layer V_layer^T`

This was intentionally distinct from Run6/7: the common component stayed **full-rank**, while only inter-layer differences were constrained to rank8/16/32.

Scope:

- all 30 decoder layers;
- all q/k/v/o/gate/up/down = **210 logical matrices**;
- 2/3/6 physical bases = 15/10/5 logical layers/base;
- activation-aware residual fitting using disjoint WikiText-2 train/test activations;
- representation-matched packed gate: Q4_GROUP64 bases + Q4_GROUP64 U/V factors versus independent Q4_GROUP64 logical matrices.

## Byte economics

The structural idea had the required order of magnitude:

- 2 bases / rank8: **10.583×** main-projection reduction;
- 2 bases / rank16: **8.561×**;
- 2 bases / rank32: **6.194×**;
- 3 bases / rank8: **7.823×**;
- 3 bases / rank16: **6.660×**;
- 3 bases / rank32: **5.134×**;
- 6 bases / rank8: **4.389×**;
- 6 bases / rank16: **3.998×**;
- 6 bases / rank32: **3.392×**.

These ratios apply only to the seven main decoder projection pools, not the entire model.

## Formal gate versus validity audit

The precommitted gate used **raw-energy-summed global operator NMSE**. By that definition six configurations formally passed because global values were ~0.019–0.021.

That gate metric was discovered to be invalid for heterogeneous operator families: summing raw numerator/denominator across sites weights operators by arbitrary absolute output energy and masks large normalized errors elsewhere. The formal output is preserved for auditability, but **is not accepted as a mechanism success**.

Normalized site evidence:

- 2 bases / rank8, **10.583×**: median site NMSE **0.5042**, only **3.33%** sites <0.05;
- 2 bases / rank32, **6.194×**: median **0.4004**;
- 3 bases / rank32, **5.134×**: median **0.3783**, only **3.81%** <0.05;
- least aggressive 6 bases / rank32, **3.392×**: median **0.3402**, only **3.81%** <0.05.

At the 5.134× point, operator medians were:

- Q **0.1291**;
- K **0.1085**;
- V **0.5630**;
- O **0.4010**;
- gate **0.3016**;
- up **0.5245**;
- down **0.6530**.

Most importantly, the **unquantized FP32 residual-factor ceilings are nearly as bad**. Example: at 3 bases/rank32 median ceiling NMSE is **0.3745** versus packed **0.3783**. At 6 bases/rank32 the ceiling is **0.3330** versus packed **0.3402**.

Therefore factor Q4 is not the primary failure. Rank≤32 inter-layer residual structure simply does not reproduce most layer functions well enough.

**Substantive decision:** `fail_current_additive_residual_sharing_ranks_le_32`.

Run-13 lesson: strong structural byte economics alone are insufficient; future structural gates must aggregate **site-normalized** errors first and include operator-family guardrails. Raw cross-operator energy-summed NMSE may not be used as the sole quality gate again.

Canonical summary: `RUN13_FINAL_STATUS.json`; workflow run `31279213965`, artifact `9027937891`, digest `sha256:465c0e157012c1d7d39d93d4baf6325a3dac72550ef211cebf8d6fa1dd6eb9e4`.

---

# Current real-model conclusion after Runs 6–8, 11, and 13

Falsified as broad post-training solutions under tested protocols:

1. universal low-rank activation projection;
2. direct whole-block post-hoc depth sharing;
3. shared low-rank output bases as the main codec;
4. naive additive residual VQ;
5. naive residual product quantization;
6. full-rank shared bases with rank≤32 post-hoc inter-layer residual adapters at the 5–15-layer sharing spans needed for extreme structural reduction.

Positive reusable findings:

- native group64-Q4 GEMV;
- native packed Q2/E4M3 latent-attention arithmetic;
- Q/K repeatedly appear more compressible than V/O/MLP;
- Hadamard rotation + block second-order error feedback materially improves W2 operator fidelity;
- rigorous real-model experiment/provenance infrastructure;
- exact evidence that structural sharing can produce 5–10× projection-byte economics, but the current post-hoc low-rank residual model cannot preserve functions.

The evidence is increasingly pointing away from **post-hoc structural conversion** and toward either a much stronger optimized W2/mixed representation or a model **trained/distilled from the outset** for parameter reuse/compressibility.

---

# Current claim boundary

> **No usable real-pretrained LARC candidate exists yet.** The ~10.5–11.8× result remains controlled/synthetic evidence only. Run 11 found a useful real operator mechanism, but not a whole-model codec. Run 13 demonstrated structurally relevant byte ratios but failed normalized held-out operator fidelity even before factor quantization. Runs 9 and 10 remain the critical external baselines.

Do not claim:

- 10× real-model compression;
- Q4_K_M parity;
- measured LARC RSS/VRAM savings;
- whole-model W2 viability from Run11;
- Run13 formal gate success as substantive viability;
- that 6.4× Run11 matrix-vs-FP16 or 5–10× Run13 projection-pool ratios are whole-model/Q4-relative reductions;
- native end-to-end LARC inference.

---

# Highest-priority work now

1. **Finish Run 9**: real Q4_K_M/Q2_K bytes, full WikiText-2 PPL, allocator pools, MaxRSS at64/2K/8K, mmap/no-mmap, throughput.
2. **Immediately execute Run 12** on the completed Run-9 artifact to determine the exact effective bpw required for 10×/20×/30× versus Q4 and the weight-vs-fixed-memory limits.
3. **Finish Run 10**: maintained optimized AutoRound W2A16G64. If promising, escalate to AutoRoundBest; if poor, stop treating more PTQ rounding alone as the likely route to 10–30×.
4. **Do not extend Run13 merely by increasing residual rank** unless byte economics are recalculated first. The least aggressive rank32 point is already only 3.39× projection reduction and median NMSE 0.34; substantially higher ranks rapidly surrender the structural gain.
5. **Primary structural research candidate if W2 is insufficient: training/distillation for compressibility**, not post-hoc fusion. Candidate architecture should learn a small set of recurrent/shared full-rank blocks plus layer/depth conditioning or adapters from the beginning, so the shared manifold is part of optimization rather than imposed afterward.
6. **Operator-adaptive low-bit work remains secondary:** Run11 supports aggressive Q/K W2 with structured rotation + second-order rounding, but V/O/up/down need materially more capacity.
7. **Runtime integration only after real quality passes:** combine any surviving weight representation with the existing packed Q2/E4M3 KV primitive and measure real RSS/TTFT/tokens/s against Run9 Q4_K_M.
8. **Standard L3/L4 evaluation:** WikiText-2, tasks/generation/rare tokens, then consumer CPU/Metal/CUDA memory and speed.

The **20–30× objective remains open**. Current evidence increasingly indicates that achieving it versus Q4 requires reducing the number of independently stored parameters—not merely quantizing the same parameter set harder—and that such reuse likely must be **learned during training/distillation rather than imposed post hoc**.
