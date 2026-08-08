# LARC Action Sheet

Canonical technical record for **LARC — Local Adaptive Representation & Compute**. Historical details remain in prior audit documents and benchmark artifacts. Artifact authority is `benchmarks/INDEX.json`; machine-readable status is `benchmarks/RUN5_FINAL_STATUS.json`.

## Objective

Reduce local-LLM **peak resident inference memory by roughly 10–30× versus a named competitive Q4-class baseline at the same context length**, while retaining useful capability. Every promoted claim must name baseline, context, byte pools, quality representation, seed coverage, and whether execution/memory is measured or modeled.

Evidence levels: **L0** format, **L1** operator/runtime, **L2** controlled model, **L2C** controlled post-training conversion, **L3** independent pretrained LLM, **L4** measured target hardware.

---

# Run 4 packed-runtime evidence retained

The upstream mainline established:

- representation-consistent single-seed L2C quality at context 64: row-Q4 teacher NLL **1.88548** vs latent-Q2/E4M3 shared NLL **1.97525**, delta **+0.08977 nats/char**, perplexity ×**1.09392**;
- native direct-packed Q2/E4M3 latent attention (`runtime/larc_q2_attention.{h,cpp}`), max abs error **2.50e-9** vs decoded reference at T=2048/rank16/head-dim32;
- packed-context structural model: **12.04×** at context64, **10.60×** at 2K, **10.50×** at 8K;
- disjoint training/selection/calibration/evaluation streams;
- deterministic basis fitting and provenance-backed generators.

Artifacts: `run4_fp8meta_l2c.json`, `run4_native_q2_attention.json`, `run4_packed_attention_context_sweep.json`.

---

# Run 5 — third external audit

Detailed response: `docs/RUN5_AUDIT_RESPONSE.md`.

## Audit finding: Run-4 full-stack ambiguity

The older `2.45371` diagnostic used Q4 weights with ordinary/full KV; it did **not** include latent-Q2 KV. No historical KV delta is added to it. Run 5 evaluates complete candidate stacks directly.

## Audit finding: correlated Q4 error was over-attributed

Six stochastic-Q4 counterfactuals gave mean depth-decorrelation benefit **+0.0343 ± 0.0482 nats/char**, with two realizations worsening. This does not establish repeated error correlation as the main mechanism.

The stronger diagnostic is weight distribution:

- independent teacher rows: absmax/RMS ~**2.13**, raw row-Q4 NMSE ~**0.73%**;
- recovered shared rows: absmax/RMS **3.10–3.24**, row-Q4 NMSE **1.56–1.73%**.

Decision: finer scale locality before dither.

Artifact: `benchmarks/run5_weight_diagnostics.json`.

## Run-5 weight representation

`Q4_GROUP64` retains the signed `[-8,7]` nibble semantics but stores one FP16 scale per contiguous <=64 weights rather than one per whole row.

Shared-model modeled payload: **79,828 B**.

### Native group64 Q4 primitive — L1

Added `Q4GroupRows` and `q4_grouped_gemv` to `runtime/larc_q4.{h,cpp}`.

Conformance test at 7×130, group size64:

- max abs error vs separately decoded arithmetic: **3.34e-6**;
- packed storage: **497 B**, exactly equal to formula;
- partial final group exercised.

Artifact: `benchmarks/run5_native_q4_group64.json`; generator/test: `tests/native_q4_group64.cpp`.

## Run-5 conversion method

Current best controlled conversion:

1. train conventional 16-independent-block teacher;
2. initialize one shared block from parameter mean;
3. **80-step teacher-layer function prefit** across all 16 teacher layer input/output transformations;
4. project matrices to group64 Q4;
5. **200-step hard-projected QAT LM recovery** at LR 1.5e-3.

Depth adapters, simple dither, and teacher-logit distillation were tested but not promoted. Function-space prefit + hard QAT was the most stable five-seed method.

## Alternate grouped-metadata KV experiment

Run 5 also tested rank16 Q2 with one FP16 scalar min/scale pair per 3-token K group and V group. It models **11.30× at context64 and 10.86× at 8K**, with five-seed mean PPL **1.047×** the project row-Q4 reference. This remains an alternate reference-only codec because no native grouped-metadata attention kernel exists.

Artifacts: `run5_memory_context.json`, `run5_fullstack_multiseed.json`.

## Preferred bridge: Run-5 weights + upstream E4M3 packed-attention codec

The five trained function-prefit/group64-QAT models were reevaluated using:

- deterministic rank16 K/V bases;
- Q4 basis storage;
- both FP16 inverse-Gram metrics;
- per-vector Q2 coefficients;
- E4M3-FN min/scale metadata;
- the same latent mathematics already validated by the native direct-packed Run-4 attention primitive.

Training seeds: `3,7,11,19,23`. Evaluation: **100,032 characters/seed**, seed999; calibration uses disjoint seed555 stream.

Against the project canonical **row-Q4 teacher** baseline:

| seed | row-Q4 teacher NLL | LARC NLL | delta nats/char | PPL ratio |
|---:|---:|---:|---:|---:|
| 3 | 1.91050 | 2.08693 | +0.17643 | 1.1930× |
| 7 | 2.27746 | 2.13375 | -0.14371 | 0.8661× |
| 11 | 2.14200 | 2.08938 | -0.05262 | 0.9487× |
| 19 | 2.15784 | 2.03713 | -0.12071 | 0.8863× |
| 23 | 2.02544 | 2.17927 | +0.15383 | 1.1663× |

Five-seed statistics:

- mean delta: **+0.00264 nats/char**;
- sample std: **0.15228**;
- mean PPL ratio: **1.01208×**;
- PPL-ratio sample std: **0.15623**;
- mean PPL ratio vs FP32 teacher: **1.33287×**.

Artifact: `benchmarks/run5_e4m3_multiseed.json`; generator: `tools/run5_e4m3_multiseed.py`.

### Combined modeled memory contract

Using group64 shared-weight bytes plus the upstream direct-packed E4M3 Q2 cache/scratch contract:

| context | modeled total reduction |
|---:|---:|
| 64 | **11.825×** |
| 256 | **11.123×** |
| 512 | **10.856×** |
| 1K | **10.682×** |
| 2K | **10.582×** |
| 4K | **10.527×** |
| 8K | **10.499×** |

Artifact: `benchmarks/run5_packed_context_sweep.json`; generator: `tools/run5_packed_context_sweep.py`.

**Quality is validated at context64 only.** Long-context rows remain structural/runtime models.

## What is and is not native now

Native L1 primitives exist separately for:

- **group64 Q4 GEMV**;
- **Q2/E4M3 latent attention**.

Run-5 quality uses mathematical/dequantized reference execution matching those storage semantics. The primitives are **not yet wired into one native full-model inference loop**, so there is no measured process-memory or full-runtime throughput result.

## Convergence status

A naive teacher continuation to 320 steps at the original constant LR degraded; this is not a convergence ceiling. Tuned/decayed multi-seed teacher/shared/smaller-model learning curves remain required.

## Transfer status

The tiny controlled model uses rank16/head-dim32 = 50%. SmolLM2 structural work uses rank16/head-dim64 = 25% and GQA; therefore the tiny-model ratio is not an upper bound on SmolLM2 arithmetic. No external-model quality exists yet.

---

# Current claim boundary after Run 5

> **Preferred controlled candidate:** teacher-layer function prefit + group64-Q4 QAT weights + rank16 Q2/E4M3 latent KV. Against the project's simple row-Q4 reference, modeled tensor residency is **11.825× lower at context64 and 10.499× lower at 8K**. Across five training seeds at context64, mean perplexity ratio is **1.012×** that same reference. Separate native L1 primitives validate the group64-Q4 GEMV and Q2/E4M3 attention arithmetic.

This is still **not** the requested final proof because:

- baseline is the project's simple row-Q4, not optimized llama.cpp Q4_K_M/IQ;
- the two native primitives are not integrated into one full-model runtime;
- memory is modeled, not measured RSS/VRAM;
- quality is synthetic character-LM, context64;
- no independent pretrained LLM has been converted;
- mean PPL remains **1.333× FP32 teacher**.

Do **not** claim 10–30× lower measured RAM/VRAM for real pretrained GGUF models.

# Highest-priority next work

1. **Integrate native primitives:** group64 packed weights + packed Q2/E4M3 attention in one inference loop; measure RSS and throughput.
2. **Real activation spectra:** first accessible pretrained Transformer, ranks 8/16/32/64/128 by projection site.
3. **Competitive baseline:** actual Q4_K_M/IQ or equivalent optimized runtime at same context/quality.
4. **Long-context quality:** validate 256→8K, not only byte accounting.
5. **Convergence study:** tuned multi-seed teacher/shared/smaller-model curves.
6. **L3:** independent pretrained 135M+ conversion, standard perplexity/tasks/rare-token evaluation.
7. **L4:** measured CUDA/Metal/CPU memory, TTFT, tokens/s.
8. **20–30×:** pursue only after 10× passes L3/L4.
