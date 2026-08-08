# LARC Action Sheet

Canonical technical record for **LARC — Local Adaptive Representation & Compute**. Historical details are retained in prior audit documents and benchmark artifacts. Artifact authority is `benchmarks/INDEX.json`; machine-readable reconciled status is `benchmarks/RUN5_FINAL_STATUS.json`.

## Objective

Reduce local-LLM **peak resident inference memory by roughly 10–30× versus a named competitive Q4-class baseline at the same context length**, while retaining useful capability. Every claim must name baseline, context, byte pools, quality delta, representation, seed coverage, and whether execution/memory is measured or modeled.

Evidence levels: **L0** format, **L1** operator/runtime, **L2** controlled trained model, **L2C** controlled post-training conversion, **L3** independent pretrained LLM, **L4** measured target hardware.

---

# Upstream Run 4 packed-runtime track — retained

## Representation-consistent single-seed L2C quality

Synthetic character LM, context 64, 100,032 final-evaluation characters; training/selection/basis-calibration/final-evaluation streams are disjoint.

| path | NLL |
|---|---:|
| independent canonical-Q4 teacher | **1.88548** |
| Q4-recovered shared model, normal KV | **1.94078** |
| shared + rank16 latent Q2 + E4M3-FN metadata + Q4 bases + K/V metrics | **1.97525** |

Total delta: **+0.08977 nats/char**, perplexity ×**1.09392**.

Artifact: `benchmarks/run4_fp8meta_l2c.json`; generator: `tools/run4_l2c_repro.py`.

## Native direct-packed latent attention — L1

`runtime/larc_q2_attention.{h,cpp}` consumes packed Q2 historical K/V, E4M3 metadata, Q4 bases and both FP16 inverse-Gram metrics without constructing FP32 historical `T×rank` arrays.

At T=2048/rank16/head-dim32:

- max abs error vs decoded reference: **2.50e-9**;
- packed cache/head: **24,576 B**;
- direct scratch/head: **8,448 B**;
- FP32 decoded latent K+V history/head: **262,144 B**.

Artifact: `benchmarks/run4_native_q2_attention.json`.

## Packed context byte model

| context | modeled reduction |
|---:|---:|
| 64 | **12.04×** |
| 256 | **11.22×** |
| 512 | **10.91×** |
| 1K | **10.71×** |
| 2K | **10.60×** |
| 4K | **10.53×** |
| 8K | **10.50×** |

Artifact: `benchmarks/run4_packed_attention_context_sweep.json`.

Quality is validated only at context 64. These are modeled tensor bytes, not measured RSS/VRAM.

---

# Run 5 — third external audit and multi-seed conversion work

Detailed audit response: `docs/RUN5_AUDIT_RESPONSE.md`.

## 1. Full-stack pairing corrected

Run-4's older `2.45371` diagnostic used Q4 weights with ordinary/full KV; it did not contain latent-Q2 KV. Run 5 therefore evaluates its grouped weight+KV stack directly rather than arithmetically adding a historical KV delta.

## 2. Depth-correlation hypothesis weakened

A six-realization stochastic-Q4 counterfactual gave mean decorrelation benefit **+0.0343 ± 0.0482 nats/char**, with two realizations worsening. This does not establish repeated-depth correlation as the main Q4 failure mechanism.

The stronger weight diagnostic is distributional:

- independent teacher rows: mean absmax/RMS ~**2.13**, raw row-Q4 NMSE ~**0.73%**;
- recovered shared rows: absmax/RMS **3.10–3.24**, raw row-Q4 NMSE **1.56–1.73%**.

Decision: use finer scale locality before depth-wise dither.

Artifact: `benchmarks/run5_weight_diagnostics.json`.

## 3. Run-5 reference weight codec

`Q4_GROUP64`: same signed `[-8,7]` code semantics as canonical Q4, but one FP16 scale per contiguous <=64 weights rather than one per whole row.

Shared-model modeled weight payload: **79,828 B** versus ~77.3 KB for full-row Q4. The small metadata increase materially reduces error on difficult shared rows.

## 4. Structural conversion fix

Current Run-5 reference conversion:

1. train conventional 16-independent-block teacher;
2. initialize one shared block from parameter mean;
3. **80-step teacher-layer function prefit** across every teacher layer's input/output transformation;
4. project matrices to group-64 Q4;
5. **200-step LM recovery at LR 1.5e-3**, hard-projecting matrices back to group-64 Q4 after every optimizer step.

Small depth adapters and teacher-logit distillation were tested but not promoted; hard-projected QAT plus function-space prefit was the most stable five-seed path.

## 5. Run-5 grouped KV reference codec

- rank16/head-dim32 latent representation;
- Q2 coefficients;
- one FP16 min+scale pair across each **3-token K group** and **3-token V group**;
- one physically shared Q4 K/V basis set for the one physical recurrent block;
- FP16 ridge-stabilized inverse-Gram for both K and V;
- incomplete current group retained as FP16 and charged as worst-case tail.

Logical KV histories and physical basis sets are counted separately: 16 logical histories, one physical basis set.

Reference workspace scales with context:

`workspace(T) = 3584 + 80T bytes`.

This is reference accounting, not a native packed scratch contract.

## 6. Grouped reference memory sweep

Baseline: **project row-Q4 teacher weights + FP16 KV + same reference workspace**; this is not llama.cpp Q4_K_M.

| context | modeled reduction |
|---:|---:|
| 64 | **11.297×** |
| 128 | **11.195×** |
| 512 | **10.986×** |
| 2K | **10.887×** |
| 8K | **10.857×** |

Artifact: `benchmarks/run5_memory_context.json`; generator: `tools/run5_memory_sweep.py`.

## 7. Five-seed representation-matched quality

Training seeds: `3,7,11,19,23`. Same seed-999 evaluation stream, **100,032 characters per seed**. The LARC path executes dequantized group-64-Q4 weights plus grouped latent-Q2 semantics, Q4 bases, both metrics and FP16 tail semantics.

Against the same project row-Q4 teacher reference used in this memory model:

- mean delta: **+0.03551 nats/char**;
- sample std: **0.16078**;
- mean perplexity ratio: **1.04705×**;
- ratio sample std: **0.17120**;
- range: **0.8969×–1.2363×**.

Against FP32 teacher, mean perplexity ratio is **1.37724×**.

Artifact: `benchmarks/run5_fullstack_multiseed.json`; protocol: `tools/run5_fullstack_protocol.py` + `tools/run5_fullstack_protocol_fp16tail.py`.

A complete post-commit five-seed one-process replay remains pending because the available execution ceiling terminates the long job; the final FP16-tail quality phase was rerun over all five retained trained models. `benchmarks/INDEX.json` records that limitation.

## 8. Evidence-track reconciliation

Run 5 does **not** supersede the packed E4M3 path.

| axis | upstream packed Run 4 | grouped Run 5 |
|---|---|---|
| controlled quality seeds | 1 | **5** |
| disjoint calibration/eval | **yes** | separate calibration/eval streams, but original training corpus differs from upstream protocol |
| weight recovery | projected row-Q4 recovery | **function prefit + group64 QAT** |
| KV metadata | E4M3 per vector | FP16 scalar per 3-token group |
| native direct-packed attention | **yes, L1** | **no** |
| context64 modeled reduction | **12.04×** | 11.30× |
| context8K modeled reduction | 10.50× | **10.86×** |
| mean multi-seed quality | not measured | **PPL ×1.047 vs project row-Q4** |

The next controlled milestone is to combine **function-prefit/group64-QAT conversion** with a **native direct-packed KV path** (E4M3 or grouped metadata) and run it across multiple seeds with disjoint calibration/evaluation.

## 9. Teacher-320 / convergence

A naive constant-LR teacher continuation degraded rather than defining a better ceiling. It is not convergence evidence. Tuned/decayed multi-seed teacher/shared/smaller-model learning curves remain open.

## 10. Tiny geometry vs SmolLM2

The tiny controlled model uses rank16/head-dim32 = 50%; SmolLM2 structural work uses rank16/head-dim64 = 25% with GQA. Therefore the tiny controlled KV ratio is not an upper bound on SmolLM2 structural arithmetic. SmolLM2 quality remains unmeasured.

---

# Current claim boundary after reconciliation

Two statements are defensible simultaneously:

1. **Native packed execution axis:** upstream Run 4 has L1 direct-packed latent-Q2 attention evidence and a single-seed L2C context-64 quality result (PPL ×1.0939) supporting a modeled **12.04×** same-context tensor contract; structural packed arithmetic stays **10.50×** at 8K without long-context quality validation.
2. **Multi-seed conversion axis:** Run 5 has five-seed representation-matched reference evidence for **11.30×→10.86× modeled tensor reduction** over context64→8K against the project's simple row-Q4 reference, with mean PPL **1.047×** that reference; this path is not yet native-packed.

Neither is the final requested proof. Do **not** claim 10–30× lower measured RAM/VRAM for real pretrained GGUF models.

# Highest-priority next work

1. **Unify the tracks:** function-prefit + group64-QAT weights with native direct-packed latent attention; test E4M3 vs grouped metadata under the same five-seed protocol.
2. **Real activation spectra:** first accessible pretrained Transformer, rank-energy/output curves at 8/16/32/64/128.
3. **Competitive Q4 baseline:** actual Q4_K_M/IQ or equivalent optimized deployment, same context and quality suite.
4. **Long-context quality:** validate 256→8K, not only byte models.
5. **Convergence:** tuned multi-seed teacher/shared/smaller-model curves.
6. **Integrated packed full-model runtime + measured RSS.**
7. **L3:** independent pretrained 135M+ conversion, standard perplexity/tasks/rare-token evaluation.
8. **L4:** measured CUDA/Metal/CPU peak memory, TTFT, tokens/s.
9. **20–30× real-model quality:** open after 10× passes L3/L4.
