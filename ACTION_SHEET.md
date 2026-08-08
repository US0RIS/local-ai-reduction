# LARC Action Sheet

Canonical technical record for **LARC — Local Adaptive Representation & Compute**. Artifact authority is `benchmarks/INDEX.json`. Current real-model summaries include `RUN6_FINAL_STATUS.json`, `RUN7_FINAL_STATUS.json`, `RUN8_FINAL_STATUS.json`, and `RUN11_FINAL_STATUS.json`. Run 5 remains historical controlled-model evidence rather than a transferable real-model architecture.

## Objective

Reduce local-LLM **peak resident inference memory by roughly 10–30× versus a named competitive Q4-class baseline at the same context length**, while retaining useful capability. Every promoted claim must name the baseline, context, byte pools, quality representation, calibration/evaluation separation, and whether execution/memory is measured or modeled.

Evidence levels: **L0** format, **L1** operator/runtime, **L2** controlled model, **L2C** controlled post-training conversion, **L3** independent pretrained LLM, **L4** measured deployment hardware.

The project is now explicitly evidence-driven: a positive operator diagnostic is not an end-to-end model result, a synthetic result cannot override a negative real-model result, and no new real-model compression claim may be promoted before a competitive external baseline exists.

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

This remains useful codec/runtime engineering evidence. It is **not** evidence that its recurrent sharing or universal low-rank assumptions transfer to pretrained LLMs.

Key artifacts: `run5_e4m3_multiseed.json`, `run5_packed_context_sweep.json`, `run5_native_q4_group64.json`, `run4_native_q2_attention.json`.

---

# Run 6 — first real pretrained falsification

Model: **`HuggingFaceTB/SmolLM2-135M`**, checkpoint `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

Run 6 attacked the two assumptions carrying most of Run 5's compression ratio: broadly low-dimensional operator geometry and aggressive cross-depth block sharing.

## Activation-aware reduced rank

49 real projection sites across layers 0/5/10/15/20/25/29 and q/k/v/o/gate/up/down:

- rank32 median held-out operator NMSE **0.28662**;
- rank32 fraction below 0.05 NMSE **8.16%**;
- rank64 median **0.24243**;
- rank64 fraction below 0.05 **20.41%**.

Precommitted gate: **`fail_low_rank_projection`**.

The failure was operator-specific. At rank128 K was strongest and Q was partially viable; V/O/MLP, especially down, remained materially worse. The conclusion is that low rank can be a local/operator-specific property but is not a universal LARC weight representation.

## Cross-depth sharing

Single neighboring-layer replacement had median PPL ratio **1.261×**. Exact contiguous sharing was severe:

- layers 8–9: **29.56× PPL**;
- layers 14–17: **5.42×**;
- layers 22–25: **5.82×**.

Layers 14–17 were then collapsed 4→1, initialized from the FP32 parameter mean, hard-projected to group64 Q4, and recovered for 24 QAT/distillation steps against a row-Q4 teacher:

- group weight reduction **3.787×**;
- whole-model modeled weight reduction from this one group **1.084×**;
- row-Q4 teacher NLL **4.96356**;
- recovered shared NLL **5.63537**;
- PPL ratio vs row-Q4 **1.95778×**.

Precommitted gate: **`fail_current_sharing_recipe`**.

**Run-6 conclusion:** direct post-hoc whole-block sharing and universal low-rank activation projection are falsified on SmolLM2-135M under the tested protocol.

Artifacts: `run6_real_model_falsification.json`, `run6_activation_aware_projection.json`, `run6_partial_real_conversion.json`, `RUN6_GATE.json`, `RUN6_FINAL_STATUS.json`.

---

# Run 7 — segmented shared-basis follow-up

Run 7 preserved layer-specific functions using like-operator shared output bases:

`W_i ≈ B_g C_i`

with one basis `B_g` per operator/depth group and unique coefficients `C_i` per logical layer. FP32 and row-Q4 representations were fitted independently; deployment ranks were chosen from held-out post-quantization operator error.

The project's simple row-Q4 baseline on the fixed slice was itself weak:

- FP32 NLL **3.88795**;
- simple row-Q4 NLL **5.09597**;
- row-Q4 PPL ratio vs FP32 **3.34683×**.

Results:

- strict gate: no group qualifies;
- balanced: six Q/K groups, whole-model modeled reduction **1.02718×**, Q4-factor PPL **2.25610×** row-Q4;
- aggressive: eight Q/K/O groups, **1.08716×** modeled reduction, PPL **3.74837×** row-Q4.

Precommitted gate: **`fail_current_shared_basis_recipe`**.

**Run-7 conclusion:** Q/K contain useful shared structure, but their byte pool is too small to drive extreme whole-model compression. The dominant MLP matrices do not tolerate this shared-basis recipe at useful ranks.

Artifacts: `run7_shared_basis_real_model.json`, `RUN7_GATE.json`, `RUN7_FINAL_STATUS.json`.

---

# Run 8 — aggressive layer-preserving vector quantization

Run 8 abandoned low-rank sharing and targeted the dominant projection byte pool while keeping every logical matrix distinct.

## Run 8A — residual VQ

Representation:

- 32-weight vectors;
- input-RMS activation weighting as a diagonal-Hessian proxy;
- FP16 per-vector magnitude;
- 16-entry FP16 residual codebooks shared by like operator / 10-layer group;
- 4-bit residual-stage indices.

At the most aggressive tested point the nominal target was **0.75 bpw** and modeled whole-model weight reduction was about **2.75×**, but NLL rose from FP32 ~**3.589** to ~**18.07**. Calibration residual energy decreased monotonically, ruling out a trivial sign/scale reconstruction failure. The 32-D 16-way dictionary simply left far too much important geometry unexplained.

Gate: **`fail_naive_additive_vq`**.

## Run 8B — residual product quantization

The 32-D direction was split into four independently coded 8-D subspaces with per-layer/per-operator codebooks.

| nominal target payload | modeled whole-weight reduction | NLL | PPL ratio vs FP32 |
|---:|---:|---:|---:|
| 1.0 bpw | **2.409×** | **16.6705** | ~479,749× |
| 1.5 bpw | **1.937×** | **19.8977** | ~12.1M× |
| 2.0 bpw | **1.620×** | **14.6485** | ~63,517× |

Even the richest RPQ point left roughly 20% activation-weighted vector residual energy for much of the model, which was still catastrophic end-to-end.

Gate: **`fail_naive_rpq`**.

**Run-8 conclusion:** naive Euclidean/diagonal-RMS vector dictionaries are not a credible primary extreme-weight codec for this model. More centroids or more greedy residual stages are not the preferred direction.

Artifacts: `run8_additive_vq_real_model.json`, `RUN8_GATE.json`, `run8_residual_pq_real_model.json`, `RUN8_RPQ_GATE.json`, `RUN8_FINAL_STATUS.json`.

---

# Run 9 — competitive llama.cpp deployment baseline — in progress

Branch/PR: `larc-run9-competitive-baseline`, PR #17.

Run 9 is intended to replace the project's weak research row-Q4 reference with an actual mature deployment target on the same SmolLM2-135M checkpoint:

- F16 GGUF quality reference;
- **Q4_K_M** named competitive Q4 baseline;
- **Q2_K** mature low-bit comparison;
- full WikiText-2 `llama perplexity` at context512;
- exact GGUF bytes and SHA256;
- GNU `time -v` process MaxRSS at context64/2048/8192, mmap and no-mmap, three repetitions;
- llama.cpp-reported model/KV/compute buffer diagnostics kept separate from process MaxRSS;
- `llama bench` prompt processing at 64/2048/8192 and tg128 after depth64/2048/8064;
- exact llama.cpp commit and runner hardware metadata.

Implementation has been adapted to current llama.cpp's unified `llama` executable. Two early workflow failures were tooling/build-interface failures, not benchmark results:

1. current upstream removed the old standalone `llama-cli` build target;
2. the unified app requires its server/CLI implementation libraries, so disabling `LLAMA_BUILD_SERVER` prevented final linkage.

Both were corrected. The latest workflow uses the unified app with required dependencies enabled. **No Run-9 numeric baseline is promoted until the completed artifact is inspected.**

Artifacts/protocol pending: `run9_llamacpp_baseline.json`, `RUN9_STATUS.json`, `docs/RUN9_COMPETITIVE_BASELINE.md`.

---

# Run 10 — maintained optimized W2A16G64 reference — in progress

Branch/PR: `larc-run10-strong-w2-reference`, PR #18.

Run 10 asks how much quality a maintained optimization-aware 2-bit PTQ system can retain before LARC invents another custom W2/sub-W2 codec.

Reference: Intel AutoRound `W2A16G64`.

Tracks:

- source SmolLM2 checkpoint under one HF evaluator;
- pure RTN W2 floor;
- tuned AutoRound W2: 200 iterations, 128 calibration samples, sequence length 2048, `enable_alg_ext=true`.

All quality comparisons use the same tokenizer and complete WikiText-2 test stream at context512. Absolute PPL is not directly substituted for llama.cpp's Run-9 PPL; within-runtime ratios are the relevant comparison.

Upstream explicitly recommends the more expensive AutoRoundBest + algorithm-extension path for maximum W2 accuracy. The current tuned run is the first serious W2 point. If it moves W2 into a useful regime, full best-quality compute is justified; if it collapses badly, that is strong evidence that the remaining 10–30× target is not a simple PTQ codec problem.

**No Run-10 numeric result is promoted until its artifact completes.**

Pending artifact: `run10_w2_reference.json`.

---

# Run 11 — second-order, rotation, and outlier W2 diagnostic

Run 11 tested a fundamentally different question from Run 8: whether real 2-bit operator fidelity improves when compression uses **block activation covariance, structured orthogonal rotation, and explicit outlier accounting** rather than only Euclidean/diagonal-RMS geometry.

Protocol:

- SmolLM2-135M checkpoint `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`;
- layers 0/5/10/15/20/25/29;
- q/k/v/o/gate/up/down = **49 real projection sites**;
- calibration: WikiText-2 raw train, 256 activation rows/site;
- held-out evaluation: disjoint WikiText-2 raw test, 64 rows/site;
- diagnostic Q2: four asymmetric levels, group64, FP16 min+scale metadata, ~2.5 bpw / ~6.4× matrix reduction vs FP16.

## Aggregate result

| representation | median held-out output NMSE | mean NMSE | fraction <0.05 | matrix reduction vs FP16 |
|---|---:|---:|---:|---:|
| plain Q2 | **0.16072** | 0.25129 | 6.12% | **6.400×** |
| block-GPTQ-style Q2 | 0.16527 | 0.23174 | 6.12% | **6.400×** |
| Hadamard Q2 | 0.15555 | 0.20998 | 6.12% | **6.396×** |
| **Hadamard + block-GPTQ Q2** | **0.09555** | **0.10649** | **36.73%** | **6.396×** |
| dense learned orthogonal Q2 ceiling | 0.15907 | 0.24625 | 6.12% | 3.740× |

The combined structured rotation + second-order error-feedback mechanism cuts median held-out output NMSE by **40.55%** relative to plain Q2 at essentially unchanged representation bytes. Neither Hadamard alone nor block-GPTQ alone produces the gain.

## Operator structure

Hadamard + block-GPTQ median NMSE by family:

- Q **0.03909**;
- K **0.02994**;
- V **0.19110**;
- O **0.14008**;
- gate **0.08547**;
- up **0.15635**;
- down **0.15705**.

This is the first positive real-model weight-compression mechanism after Runs 6–8, but it is **not a usable whole-model W2 codec**. Q/K look plausible for aggressive W2. The dominant V/O/up/down pools remain too inaccurate.

## Outlier and learned-rotation controls

Simple activation-weighted FP16 outlier escape did not improve the best combined error/byte point:

- no escape: median NMSE **0.09555**, ~6.396×;
- +1% FP16 columns: **0.10671**, ~5.995×;
- +2%: **0.10686**, ~5.642×;
- +5%: **0.10414**, ~4.836×.

Therefore this escape strategy is not promoted.

The dense covariance-eigenvector rotation ceiling also failed to justify its representation cost: median NMSE **0.15907**, only **3.740×** vs FP16 after transform bytes. The positive result is specifically the cheap structured Hadamard + covariance-aware rounding interaction.

## Run-11 decision

**Promote as a component mechanism:** structured Hadamard rotation + block second-order/GPTQ-style rounding/error feedback.

**Do not promote as an end-to-end codec:** only 36.7% of sites pass 0.05 NMSE; V/O/up/down remain weak.

**Do not promote:** tested dense learned rotation or 1–5% FP16 outlier escape.

Canonical summary: `RUN11_FINAL_STATUS.json`. Generator: `tools/run11_second_order_diagnostic.py`. Workflow run `31278590079`, artifact `9027744924`, digest `sha256:ef4145385d3ed5ce7761ba8ead540226bb0f3e9f85fe79cb15d4f61b83ee2192`.

---

# Current real-model conclusion after Runs 6–8 and 11

The following mechanisms are falsified as **broad post-training solutions** on SmolLM2-135M under the tested protocols:

1. universal low-rank activation projection;
2. direct whole-block cross-depth sharing;
3. segmented shared output bases as the main weight codec;
4. naive sub-2-bit residual vector quantization;
5. naive residual product quantization.

Run 11 adds an important positive qualification:

- **structured Hadamard rotation + second-order block error feedback materially improves real W2 operator fidelity without a dense-transform byte penalty**;
- Q/K are consistently easier across Runs 6, 7, and 11;
- the dominant MLP/V/O byte pools remain the compression bottleneck.

The strongest reusable components are now:

- native group64-Q4 weight GEMV;
- native packed Q2/E4M3 latent-attention arithmetic;
- real-model evidence that Q/K can tolerate substantially more aggressive representation than MLP/V/O;
- Run-11 evidence for structured rotation + covariance-aware rounding;
- the experiment/provenance infrastructure for representation-matched real-model testing.

The Run-5 recurrent/shared architecture must **not** be revived as though real transfer merely needs more tuning.

---

# Competitive-baseline gap

The project still lacks a **completed committed** optimized deployment baseline. This blocks any new real-model compression claim even though Run 11 found a positive mechanism.

Run 9 must establish actual llama.cpp Q4_K_M/Q2_K:

- exact GGUF bytes/hashes;
- full WikiText-2 PPL;
- MaxRSS at 64/2K/8K;
- mmap/non-mmap;
- llama.cpp allocator pools;
- PP/TG throughput;
- exact runtime/hardware provenance.

Run 10 must independently establish what competent W2A16G64 optimization can retain.

Only after those two results are known can a LARC representation be judged against the real frontier rather than the project's weak row-Q4 research reference.

---

# Current claim boundary

> **No usable real-pretrained LARC candidate exists yet.** The ~10.5–11.8× result remains controlled/synthetic evidence only. Runs 6–8 falsified the mechanisms responsible for most of that ratio as broad post-hoc real-model techniques. Run 11 found a genuine positive operator-level mechanism—Hadamard rotation plus second-order error feedback—but it has not passed end-to-end quality and does not solve the dominant MLP/V/O operators.

Do not claim:

- 10× real-model compression;
- Q4_K_M parity;
- measured LARC RSS/VRAM savings;
- whole-model W2 viability from Run 11;
- that a 6.396× **matrix-vs-FP16** Run-11 ratio is a whole-model or Q4-relative compression ratio;
- native end-to-end LARC inference;
- that naive low-rank/shared/VQ mechanisms remain preferred.

---

# Highest-priority work now

1. **Finish Run 9:** actual current llama.cpp Q4_K_M/Q2_K deployment baseline at 64/2K/8K, including WikiText-2 PPL, process MaxRSS, allocator pools, and throughput.
2. **Finish Run 10:** tuned AutoRound W2A16G64 on the same checkpoint/corpus. If tuned W2 is promising, escalate to AutoRoundBest/maximum-quality W2 rather than guessing at the PTQ frontier.
3. **Build the first operator-adaptive end-to-end candidate only after 1–2:** Run 11 supports aggressive Hadamard+second-order W2 for Q/K, but V/O/up/down must retain more capacity. Test mixed precision/optimized rounding with exact bytes and end-to-end WikiText quality. No promotion from operator NMSE alone.
4. **Use second-order/rotation insight, not more naive dictionaries:** if custom PTQ continues, candidate mechanisms are structured rotations, Hessian-aware/discrete rounding, operator-specific bit allocation, and training-aware recovery. The tested simple FP16 outlier escape is not preferred.
5. **Escalate to a learned/distilled architecture path if competent W2 remains far from useful quality:** >10× versus Q4 may be a training/distillation problem rather than a post-training codec problem. A model trained from the outset for recurrent/shared/dictionary/low-bit structure remains a distinct open hypothesis even though direct post-hoc sharing failed.
6. **Runtime integration:** once a real weight representation passes end-to-end quality, integrate it with the existing packed Q2/E4M3 KV primitive and measure actual RSS/TTFT/tokens/s under the same protocol as Q4_K_M.
7. **Standard evaluation:** real candidates must survive WikiText-2 plus task/generation/rare-token tests before L3 promotion.
8. **L4 hardware:** CUDA/Metal/consumer-CPU measurements only become decisive after a real L3 representation survives quality.

The **20–30× objective remains open**. It must be rebuilt from real-model evidence rather than extrapolated from synthetic ratios.
