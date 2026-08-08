# LARC Action Sheet

This is the persistent technical project record for **LARC — Local Adaptive Representation & Compute**. Update it on every substantive research/implementation run. Compression claims are intentionally split by evidence level so synthetic/operator results cannot be confused with pretrained-model or hardware results.

## Project objective

Create a local-AI storage/execution standard that makes capable language models practical on materially weaker hardware than ordinary GGUF deployment.

Initial target: **10–30× lower model/storage and peak resident inference memory than a Q4-class GGUF baseline**, while retaining useful model capability and avoiding a runtime that reconstructs full dense weights.

### Hard success criteria

A complete result must report:

1. full file bytes,
2. unique resident weight bytes,
3. KV-cache bytes at a stated context length,
4. bounded scratch/workspace,
5. peak total resident inference bytes,
6. retained model quality,
7. inference speed/latency,
8. baseline name/configuration,
9. evidence class: synthetic, modeled, trained conformance model, independent pretrained model, or measured target hardware.

---

# Run 1 — 2026-08-08

## Status

Established the initial LARC direction and proved that structural representations can mechanically enter the requested storage range. **Run 1 did not prove real-LLM quality or VRAM reduction.**

## Starting repository state

The repository contained only a one-line README. No implementation or benchmark artifacts existed.

## v0.1 design decision

LARC is an execution representation, not merely a smaller wrapper around dense Q2/Q4 tensors. Initial logical representation:

`activation-subspace core + progressive residual refinements + sensitive-tensor fallbacks`.

GGUF/SafeTensors are import/interchange sources rather than the internal runtime contract.

## HRVQ64 experiment

Implemented 64-weight additive vector coding with shared RMS scales and progressive 256-entry codebooks.

Nominal payload rates:

| Stages | Nominal bpw | Ideal multiple vs 4.5 bpw |
|---:|---:|---:|
| 1 | 0.1875 | 24.0× |
| 2 | 0.3125 | 14.4× |
| 3 | 0.4375 | 10.29× |

Synthetic quality was poor at these rates. Three-stage output NMSE was approximately 0.67 (Gaussian), 0.61 (heavy-tail), and 0.49 (low-rank-plus-noise). **Decision:** HRVQ is not the base representation; retain it only as a lower-importance residual/refinement candidate.

Artifact: `benchmarks/benchmark_hrvq_run1.json`.

## Activation-subspace projection bundles

For multiple operators consuming a shared activation domain, store shared `U`/`B` plus projected operators rather than independent full matrices.

Synthetic benchmark, five 384×384 operators:

| Core | Factor | Reduction vs row-Q4 | held-out output NMSE (98% activation energy in core) |
|---|---:|---:|---:|
| rank 10 (2.60%) | Q4 | **23.10×** | 0.0260 |
| rank 10 (2.60%) | Q8 | **13.47×** | 0.0200 |
| rank 19 (4.95%) | Q4 | **13.47×** | 0.0280 |

Artifact: `benchmarks/benchmark_projection_run1.json`.

## Run 1 conclusion

The requested byte range is structurally possible when activation energy is concentrated, but weight-only sub-0.5-bpw vector coding is too destructive. Primary direction became **activation-aware shared structure + progressive residuals**.

---

# Run 2 — 2026-08-08

## Goal

Stop counting small files alone. Attack the other original requirements directly:

- real execution from compressed structures,
- weight residency,
- KV residency,
- total inference memory,
- quality retention,
- random-access/paged format semantics,
- real/pretrained-model validation harness.

## Evidence levels introduced in LARC v0.2

- **L0 Structural:** container/codec integrity and byte accounting.
- **L1 Operator:** compressed-domain kernels and held-out operator error.
- **L2 Conformance model:** a trained autoregressive language model exercises the storage/runtime mechanisms and passes memory/quality gates.
- **L3 External pretrained model:** post-training conversion of an independently pretrained LLM, compared at equal context against a named GGUF baseline.
- **L4 Hardware:** measured peak RAM/VRAM and throughput on target hardware.

This distinction is mandatory in future reports.

## A. Direct packed-Q4 CPU execution — ACHIEVED at L1

Implemented:

- `runtime/larc_q4.h`
- `runtime/larc_q4.cpp`
- Python reference `larc/q4_runtime.py`
- reproducibility tests `tests/native_q4_smoke.cpp` and `tests/native_q4_bench.cpp`

Kernel contract:

`z = Bx`, then `y = Az`

Both `B` and `A` remain packed INT4. Nibbles are decoded inside the dot product. No dense FP16/FP32 `W = AB` is constructed. Scratch is rank-sized.

### Correctness smoke

Rank-19 projected GEMV:

- maximum absolute error against separately dequantized reference: **~2.29e-5**
- temporary rank scratch: **76 bytes**

### Transformer-like CPU microbenchmark

Shape: 1536×576, rank 32, compiled with `g++ -O3 -march=native -std=c++17`.

| Metric | Direct row-Q4 | LARC factors |
|---|---:|---:|
| resident weight bytes | 448,512 | 40,064 |
| resident reduction | — | **11.19×** |
| GEMV time | 438.9 µs | 33.1 µs |
| speed | 1× | **13.27×** |
| rank scratch | — | **128 B** |

Output NMSE versus the independently row-Q4 dense synthetic operator was ~0.0462; that number measures the synthetic factorization, not kernel arithmetic error.

Artifact: `benchmarks/run2_native_q4_kernel.json`.

## B. GPU compressed-domain contract — IMPLEMENTED SOURCE, NOT L4 VALIDATED

Added `runtime/triton_q4.py`.

The Triton kernel reads packed nibbles directly and supports projected `A(Bx)` with rank-sized CUDA scratch. The current environment has no CUDA/Triton hardware path, therefore:

- source/interface: implemented,
- syntax/import boundary: implemented,
- measured GPU correctness: **not achieved**, 
- measured GPU VRAM: **not achieved**,
- measured GPU speed: **not achieved**.

Do not count this as an L4 result.

## C. Recursive/shared operator graph — ACHIEVED at L2 representation level

The v0.2 manifest now permits multiple logical layers to reference one physical block bundle, with optional future depth-specific adapters. This follows the practical direction of modern recursive/shared-weight Transformer research rather than assuming every logical layer must own an independent dense tensor.

### Exact-equivalence recurrent conformance test

A trained 16-logical-layer language model using one physical Transformer block was compared against an explicitly duplicated 16-copy representation of the same logical function.

Results:

- shared validation NLL: 0.6147951
- duplicated validation NLL: 0.6147951
- max logit difference: **0.0**
- FP32 unique weight residency reduction: **13.552×**
- Q4-style logical file reduction: **13.404×**
- exact quality equivalence: **yes**

Artifact: `benchmarks/run2_recurrent_conformance.json`.

This proves shared logical/physical graph semantics, not broad model intelligence.

## D. Latent 2-bit KV cache

Implemented `larc/latent_kv.py`.

K/V are projected to learned latent bases. Historical full-dimensional K/V vectors do not need to remain resident. Attention projects the current query into latent-key space and reconstructs only the weighted value aggregate.

### D1. Initial per-token/per-token Q2 codec

Controlled low-rank attention simulation, head dimension 64, latent rank 16:

- K NMSE: ~0.1245
- V NMSE: ~0.1222
- attention-output NMSE: **0.00280**
- modeled SmolLM2 KV reduction at 2K: **15.52×**
- modeled SmolLM2 KV reduction at 8K: **15.88×**

Artifact: `benchmarks/run2_latent_kv_synthetic.json`.

### D2. KIVI-oriented latent Q2 codec

Updated the representation to use the empirically motivated asymmetry:

- keys: per latent channel over token groups,
- values: per token,
- coefficients: 2-bit asymmetric,
- bases: Q4/Q8 eligible.

Controlled rank-16 result:

- attention-output NMSE: **0.00831**
- modeled SmolLM2 KV reduction at 2K: **18.96×**
- modeled SmolLM2 KV reduction at 8K: **19.50×**

Artifact: `benchmarks/run2_kivi_latent_kv_synthetic.json`.

The attention synthetic is not a real-LLM quality result.

## E. End-to-end trained conformance model — FIRST COMBINED MEMORY + QUALITY PASS

Implemented `tools/recurrent_kv_endtoend.py`.

Configuration:

- autoregressive character language model,
- hidden width 128,
- four heads,
- 16 logical recursive depths,
- context 64,
- latent KV rank 12,
- one physical Transformer block,
- actual packed uint8 2-bit KV coefficient tensors with FP16 quantization metadata,
- baseline token-by-token inference independently checked against full causal inference.

### Runtime verification

Maximum baseline incremental-vs-full logit error: **1.335e-5**.

### Quality

Held-out generated-story corpus:

- baseline NLL: **2.01662**
- LARC latent-Q2 NLL: **2.24911**
- NLL increase: **11.53%**
- predefined screening gate: ≤15%
- gate result: **PASS**

### Actual representation bytes

| Pool | Baseline | LARC | Reduction |
|---|---:|---:|---:|
| Q4 logical/weight payload | 1,129,482 B | 77,322 B | **14.61×** |
| KV payload | 524,288 B FP16 | 57,344 B packed Q2 + 1,728 B shared basis | **8.88×** |
| bounded scratch | 7,680 B | 7,680 B | 1× |
| **total** | **1,661,450 B** | **144,074 B** | **11.53×** |

Hard minimum total-memory target: ≥10×. Result: **PASS at L2**.

Artifact: `benchmarks/run2_recurrent_kv_endtoend.json`.

### Interpretation

This is the first executable trained language-model test where LARC simultaneously passes:

- >10× weight storage/residency,
- >10× total inference-tensor memory,
- bounded context held constant,
- predefined quality-degradation gate.

It is intentionally a small LARC-native conformance model trained on a controlled story corpus. It **does not establish that an arbitrary pretrained 135M/1B/7B model retains comparable intelligence after 10× conversion**.

## F. SmolLM2-135M full-size memory plan — MODELED, NOT L3/L4

Target baseline: published SmolLM2-135M Q4_K_M reference size ~105 MB. LARC profile sizes account for Q4 projection factors, norms, latent KV and bounded workspace.

With KIVI-style latent-Q2 rank 16:

| Profile | Context | LARC weight bytes | Weight reduction vs 105 MB | KV reduction | Modeled total-memory reduction |
|---|---:|---:|---:|---:|---:|
| 10x | 2K | 8.06 MB | 13.02× | 18.96× | **14.00×** |
| 10x | 8K | 8.06 MB | 13.02× | 19.50× | **16.26×** |
| 15x | 2K | 5.31 MB | 19.78× | 18.96× | **18.73×** |
| 20x | 2K | 3.43 MB | 30.60× | 18.96× | **24.34×** |
| 30x | 2K | 2.43 MB | 43.23× | 18.96× | **28.98×** |

Artifacts:

- `benchmarks/run2_kivi_memory_plan_rank16.json`
- `tools/memory_plan.py`

These numbers demonstrate that the designed structures fit the 10–30× total-memory envelope mathematically. They are **not measured SmolLM2 VRAM and contain no SmolLM2 quality evidence**.

## G. Real external pretrained-model harness — IMPLEMENTED, L3 BLOCKED

Added `tools/real_model_benchmark.py` targeting `HuggingFaceTB/SmolLM2-135M`, with profiles for 10× / 15× / 20× / 30×.

The harness is designed to:

- download the independent pretrained model,
- measure baseline held-out NLL/perplexity,
- capture calibration activations,
- replace compatible operators with shared projection factors,
- factor the tied vocabulary matrix,
- execute through packed low-bit runtime modules,
- report complete encoded weight bytes,
- measure post-conversion NLL/perplexity and generation samples.

A GitHub Actions matrix was also added to use a hosted runner because the local environment cannot fetch Hugging Face/Xet model weights.

### Infrastructure failure, not codec failure

The hosted workflow jobs failed **before any workflow step began**, including after replacing reusable `uses:` actions with a pure `run:` workflow. No checkpoint was downloaded, no conversion ran, and no scientific result was produced.

Direct raw GitHub/Hugging Face model-binary retrieval is also blocked by the current execution network boundary.

Therefore L3 remains **unpassed**, not failed experimentally.

## H. v0.2 paged container — ACHIEVED at L0

Implemented `larc/paged_container.py` and `tests/test_paged_container.py`.

Properties:

- fixed 64-byte header,
- fixed 64-byte page records,
- 4 KiB default payload alignment,
- stable numeric research codec IDs,
- per-page CRC32,
- dependency groups,
- `REQUIRED`, `SHARED`, `REFINEMENT`, `STREAMABLE`, `KV_BASIS` flags,
- mmap random-access page views,
- unique residency accounting for shared references.

Local round-trip test passed: page offsets were 4 KiB aligned, CRC verified, mmap views matched original payload, and duplicate page references were counted once.

## I. Research-direction changes from Run 2

### Promoted

1. **Recursive/shared physical bundles** — real modern prior art shows parameter-sharing models can retain substantial capability and may be converted from pretrained Transformers with adaptation/uptraining.
2. **Projection factors** — remain useful both within and across shared bundles.
3. **Latent low-bit KV** — necessary because KV becomes dominant after weight storage falls by an order of magnitude.
4. **Direct packed kernels** — mandatory, not optional optimization.
5. **Paged/resident-budget runtime** — required to distinguish file size from actual device memory.

### Deprioritized

- whole-model sub-0.5-bpw raw-weight vector coding as the primary representation,
- any workflow that decompresses a `.larc` model to a conventional dense model before execution.

## J. Original goal status after Run 2

| Original requirement | L2 status | L3/L4 status |
|---|---|---|
| 10–30× smaller model representation | **PASS: 14.61× on trained recurrent conformance model** | **Not yet proven on independent pretrained LLM** |
| 10–30× less resident weight memory | **PASS: 14.61× conformance; 11.19× native operator microbenchmark** | Not measured on external pretrained GPU runtime |
| 10–30× less total inference memory | **PASS: 11.53× conformance model** | SmolLM2 only modeled 14–29×; GPU VRAM not measured |
| comparable/reasonable quality | **Screening PASS: +11.53% NLL on controlled trained model** | Broad/pretrained intelligence not established |
| compressed-domain CPU kernels | **PASS, measured** | — |
| compressed-domain GPU kernels | source contract implemented | **Hardware validation not achieved** |
| random-access standard/runtime format | **PASS at research v0.2/L0** | production ABI not frozen |

## K. What still must happen before claiming the full user-level goal

The project MUST NOT be described as having made arbitrary local LLMs use 10–30× less VRAM yet. The remaining decisive milestones are:

### P0 — L3 independent pretrained model

1. Obtain an accessible independent pretrained checkpoint (SmolLM2-135M remains preferred first target).
2. Run baseline Q4_K_M at the same context and record quality/memory.
3. Convert tensor-by-tensor without requiring a second full dense local copy.
4. Sweep projection/shared/adaptor/residual profiles at 10×, 15×, 20×, 30×.
5. Evaluate perplexity plus external tasks/generation, not only calibration-domain NLL.
6. Reject profiles that cross the agreed quality threshold.

### P0 — Real latent-KV LLM validation

- capture per-layer/head K/V distributions from the target pretrained model,
- fit latent bases on calibration data,
- evaluate KIVI-style latent Q2 at long context,
- implement packed latent attention kernels so full historical K/V are never reconstructed.

### P0 — L4 target hardware

- run CPU baseline vs LARC with peak RSS and tokens/s,
- run CUDA/Triton on an NVIDIA GPU and record peak allocated/reserved VRAM,
- run Metal on Apple Silicon or add a native Metal packed kernel,
- compare at identical context and generation settings.

### P1 — pretrained recursive conversion

Use modern relaxed-recursive / cross-layer-sharing methods as the conversion route when independent layers do not factor sufficiently. Candidate path:

`pretrained full stack → shared physical block groups → depth-wise low-rank adapters → short calibration/uptraining/distillation → low-bit factors`.

This is likely more realistic for 10×+ quality retention than forcing every unrelated pretrained matrix into an ultra-low-rank approximation.

## Run 2 repository artifacts

Key new artifacts:

- `docs/SPEC.md` — LARC v0.2 specification.
- `larc/paged_container.py` — mmap paged format.
- `larc/q4_runtime.py` — Python packed-Q4 execution reference.
- `runtime/larc_q4.{h,cpp}` — native packed-domain CPU kernel.
- `runtime/triton_q4.py` — CUDA/Triton packed-domain reference kernel.
- `larc/latent_kv.py` — latent Q2 and KIVI-oriented latent KV codecs.
- `tools/recurrent_kv_endtoend.py` — combined L2 memory/quality test.
- `tools/real_model_benchmark.py` — L3 SmolLM2 harness.
- `tools/memory_plan.py` — complete memory-pool accounting.
- `benchmarks/run2_recurrent_kv_endtoend.json` — first combined >10× memory + quality pass.
- `benchmarks/run2_native_q4_kernel.json` — CPU kernel measurements.
- `benchmarks/run2_kivi_latent_kv_synthetic.json` — asymmetric latent KV study.
- `benchmarks/run2_kivi_memory_plan_rank16.json` — SmolLM2-shaped memory accounting.
- `benchmarks/run2_recurrent_conformance.json` — exact shared-graph equivalence test.

## Run 2 conclusion

**A LARC-native trained language model has now crossed the 10× total-memory gate while staying inside the predefined quality screening gate, and the direct packed CPU execution path is measured and working.** This resolves the earlier question of whether the standard/runtime architecture can in principle satisfy the complete memory objective.

**The project has not yet crossed the independent-pretrained-model or measured-GPU gates.** Those are now the controlling milestones. Any statement that LARC already provides 10–30× lower VRAM for arbitrary GGUF models would be unsupported by the current evidence.
