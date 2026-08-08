# LARC Action Sheet

Canonical technical record for **LARC — Local Adaptive Representation & Compute**. Detailed historical corrections are retained in `docs/RUN3_AUDIT_CORRECTIONS.md`, `docs/RUN5_AUDIT_RESPONSE.md`, git history, and the machine-readable artifacts under `benchmarks/`.

## Objective

Reduce **complete peak resident local-LLM inference memory** by roughly **10–30× versus a named competitive Q4-class baseline at the same context length**, while retaining useful capability and avoiding dense weight reconstruction.

Every promoted result must state:

- exact baseline representation/runtime;
- context length;
- weight bytes;
- KV bytes;
- workspace/scratch;
- whether memory is modeled or measured;
- quality metric and evaluation tokens;
- whether quality executes exactly the representation whose bytes are charged;
- evidence level and seed distribution.

Evidence levels:

- **L0** — format/container integrity;
- **L1** — operator/kernel evidence;
- **L2** — controlled trained-model evidence;
- **L2C** — conventional independent-layer model trained first, then converted/recovered;
- **L3** — independently pretrained external LLM;
- **L4** — measured target-hardware RAM/VRAM + throughput.

A lower evidence level must never be promoted as a higher one.

---

# Historical findings that remain valid

## L0 paged container

Implemented random-access `.larc` research container with 64-byte header/page records, alignment, mmap views, CRC32, dependency groups, shared/refinement flags, and selected-page byte accounting.

## HRVQ64

Nominal 1/2/3-stage payload rates were 0.1875/0.3125/0.4375 bpw excluding codebook transmission. At 0.4375 bpw the Gaussian squared-error rate-distortion lower bound is ~0.545 NMSE; observed Gaussian NMSE ~0.672. Main conclusion: sub-0.5-bpw unstructured weight coding is fundamentally too low-rate for high fidelity. Codebook bytes must always be included unless globally amortized.

## Activation-subspace projection

Synthetic constructed-covariance tests showed 13–23× operator storage can retain low output error **conditional on** input activations truly occupying tiny subspaces. Quantize-first calibration-weighted fitting improved the rank-10/Q4/98%-energy synthetic point to ~0.0248 output NMSE. This remains conditional synthetic evidence only.

## Packed Q4 execution

Native CPU kernel consumes packed nibbles directly and computes projected `A(Bx)` without reconstructing dense `W`. Canonical Q4 uses signed `[-8,7]`, code=`q+8`, low nibble first, FP16 scale, range-aware scale `max(max_positive/7, max_negative_abs/8)`.

The old Run-2 operator artifact reporting 0.046151 NMSE was revoked as unreproducible. A redesigned low-noise rank-32 benchmark with theoretical truncation floor 0.002299 measured ~0.03330 NMSE vs exact FP32 at 12.062× factor payload reduction, showing factor quantization itself is material.

## Recursive physical aliasing

One physical block invoked at 16 logical depths is exactly equivalent to 16 literal copies **only when those copies contain identical learned parameters**. That is lossless representation aliasing, not evidence that arbitrary independently trained layers can be collapsed without recovery.

## Run-3/Run-4 provenance corrections

Several prior headline artifacts were downgraded or superseded because evaluation aggregation, basis execution, weight representation, or generator provenance did not match the claimed accounting. `benchmarks/INDEX.json` is now the provenance registry. Current promotion rules require committed generators, exact context/baseline labels, and representation-matched memory/quality.

---

# Run 5 — third external audit response

Detailed rationale: `docs/RUN5_AUDIT_RESPONSE.md`.

## 1. Run-4 complete-stack ambiguity — RESOLVED

Run-4 converted-model NLL `2.45371` used row-Q4-dequantized weights with ordinary/full KV. It did **not** include latent-Q2 KV. Therefore no Run-3 KV delta is arithmetically added to it. Run 5 evaluates the full weight+KV representation jointly.

## 2. Depth-correlated Q4 error — NOT THE MAIN SUPPORTED MECHANISM

A six-realization counterfactual stochastic-Q4 experiment compared one error realization reused at all depths with independent error realizations at each invocation.

Mean decorrelation benefit:

- **+0.03432 nats/char**;
- sample std **0.04824**;
- sign reversed in two realizations.

This is weak, noisy evidence. It does not justify treating correlation as the principal cause of recurrent Q4 degradation.

More decisive row statistics across seeds 7/11/19/23:

- teacher block mean absmax/RMS: ~**2.13**;
- shared recovered block: **3.10–3.24**;
- teacher raw row-Q4 matrix NMSE: ~**0.73%**;
- shared block: **1.56–1.73%**.

**Current interpretation:** the shared block's weight distribution is substantially harder for one-scale-per-row Q4. Finer scale locality is better supported than depth dither.

Artifact: `benchmarks/run5_weight_diagnostics.json`.

## 3. Weight codec — GROUP-64 Q4 SELECTED FOR CONTROLLED CANDIDATE

The LARC controlled candidate uses one FP16 scale per **64-weight sub-row group**, while retaining signed Q4 codes.

Shared-model modeled weight payload:

- old row-Q4: ~77.3 KB;
- group-64 Q4: **79,828 B**.

This small metadata increase materially reduces quantization error on difficult shared weights.

## 4. Conversion/recovery — FUNCTION PREFIT + HARD-PROJECTED QAT

Parameter averaging alone is a poor 16→1 collapse initialization.

Current conversion sequence:

1. train the conventional **16-independent-block teacher** for 120 steps;
2. initialize one shared block from the parameter mean;
3. **80-step teacher-layer function prefit**: collect every teacher layer's `(input, output)` transformation and train the one shared block by MSE across all 16 roles;
4. project all matrix weights to the exact group-64 Q4 representation;
5. **200-step LM recovery at LR 1.5e-3**, hard-projecting matrix weights back to group-64 Q4 after every optimizer step.

Tested alternatives not promoted:

- rank-2/4/8 depth adapters: seed-dependent and no five-seed mean gain under tested schedule;
- teacher-logit distillation: did not improve the actual quantized full stack enough;
- dither: weak mechanism signal;
- plain QAT without function prefit: improved robustness but left larger structural loss.

## 5. KV codec — GROUP-3 LATENT Q2 SELECTED

Controlled geometry:

- 16 logical KV histories;
- d_head=32;
- latent rank=16;
- Q2 coefficients;
- one FP16 min+scale pair across each **3-token K group** and each **3-token V group**;
- one physically shared Q4 K/V basis set because the model has one physical recurrent block;
- FP16 inverse-Gram matrices for **both K and V**;
- ridge rule: `1e-5 * mean(diag(B B^T))` per head;
- incomplete current group retained as **FP16 residual tail** and explicitly charged.

Larger metadata groups save more bytes but degrade quality. Group 3 is the current controlled rate-distortion compromise that keeps modeled total memory above 10× through 8K context.

## 6. Context-dependent workspace — CHARGED

Reference controlled workspace:

`workspace(T) = (D + D + 3D + FF + H*T + rank*T) * 4`

For D=128, H=4, FF=256, rank=16:

`workspace(T) = 3584 + 80T bytes`.

Examples:

- context 64: 8,704 B;
- 2K: 167,424 B;
- 8K: 658,944 B.

This is still a reference tensor-accounting model, **not a measured allocator/RSS trace**.

## 7. Current controlled memory result

Baseline: **project canonical row-Q4 teacher weights (1,129,482 B) + FP16 KV + the same reference workspace**.

LARC shared group-64 Q4 weight payload: **79,828 B**.

| context | modeled total reduction |
|---:|---:|
| 64 | **11.297×** |
| 128 | **11.195×** |
| 256 | **11.072×** |
| 512 | **10.986×** |
| 1K | **10.922×** |
| 2K | **10.887×** |
| 4K | **10.867×** |
| 8K | **10.857×** |

Artifact: `benchmarks/run5_memory_context.json`.

This supersedes the controlled Run-4 ~8× long-context result for the selected grouped-KV candidate.

## 8. Current controlled full-stack quality — FIVE SEEDS

Training seeds: `3, 7, 11, 19, 23`.

Evaluation stream seed: `999`.

Evaluation size: **100,032 characters per seed**.

Quality executes:

- the group-64 Q4 LARC weights whose bytes are charged;
- group-3 latent-Q2 KV;
- Q4 K/V bases;
- K/V inverse-Grams;
- FP16 metadata;
- FP16 current-group tail semantics.

Baseline quality is the **same project row-Q4 teacher representation used in memory accounting**, with ordinary/full KV.

| seed | row-Q4 teacher NLL | LARC full-stack NLL | delta nats/char | PPL ratio |
|---:|---:|---:|---:|---:|
| 3 | 1.91050 | 2.12264 | +0.21214 | 1.2363× |
| 7 | 2.27746 | 2.18102 | -0.09644 | 0.9081× |
| 11 | 2.14200 | 2.10687 | -0.03513 | 0.9655× |
| 19 | 2.15784 | 2.04906 | -0.10878 | 0.8969× |
| 23 | 2.02544 | 2.23119 | +0.20575 | 1.2284× |

Five-seed statistics vs project row-Q4 baseline:

- mean delta: **+0.03551 nats/char**;
- sample std: **0.16078 nats/char**;
- mean PPL ratio: **1.04705×**;
- PPL-ratio sample std: **0.17120**;
- range: **0.8969×–1.2363×**.

Absolute comparison vs FP32 teacher:

- mean delta: **+0.31938 nats/char**;
- mean PPL ratio: **1.37724×**.

Artifact: `benchmarks/run5_fullstack_multiseed.json`.

### Interpretation

The controlled ≥10× gate is **re-established at L2C only against the project's simple row-Q4 baseline**. On average across five seeds, the full LARC representation is ~4.7% higher perplexity than that baseline while modeled tensor memory is 10.86–11.30× smaller over 64–8K context.

This is **not** a claim of parity with llama.cpp Q4_K_M. The project row-Q4 baseline is relatively weak; LARC is still ~1.377× the FP32 teacher perplexity on average.

## 9. Provenance

- `tools/run5_memory_sweep.py` generates the context accounting;
- `tools/run5_fullstack_protocol.py` contains the self-contained five-seed training/conversion/evaluation protocol;
- `tools/run5_fullstack_protocol_fp16tail.py` enforces the exact FP16 incomplete-group semantics used by the promoted quality result;
- `benchmarks/INDEX.json` records replay status.

The final quality phase was rerun over all five retained trained models after the FP16-tail fix. A complete five-seed one-process replay of the committed generator has **not independently completed after commit** because the execution ceiling terminates the long job. This is explicitly recorded rather than labeled CI-reproduced.

## 10. Convergence control

A naive teacher-320 run at the original constant LR degraded rather than providing a better ceiling. It is not promoted as a convergence result. Required next control: tuned/decayed schedules and learning curves for teacher, converted/shared, and smaller-model baselines.

## 11. SmolLM2 transfer note

The controlled tiny model uses rank 16 / head_dim 32 = 50% latent rank. SmolLM2's planner uses rank 16 / head_dim 64 = 25% plus GQA. Therefore the old 8× controlled asymptote was **not** an upper bound on SmolLM2's structural KV ratio. SmolLM2's 18–19× KV figures remain arithmetic only until real-model quality exists.

---

# Current evidence status after Run 5

| requirement | status |
|---|---|
| paged random-access format | **L0 implemented** |
| canonical packed Q4 CPU primitive | **L1 implemented** |
| controlled >10× modeled tensor memory across 64–8K | **L2C PASS vs project row-Q4 baseline** |
| controlled quality at same representation | **five-seed mean PPL ×1.047 vs project row-Q4; PASS as screening evidence** |
| parity vs optimized Q4_K_M/IQ | **OPEN** |
| measured process RAM | **OPEN** |
| independent pretrained 135M+ model | **L3 OPEN** |
| real activation-spectrum support for aggressive ranks | **OPEN / decisive transfer gate** |
| CUDA/Metal measured VRAM + throughput | **L4 OPEN** |
| 20–30× retained-quality real-model result | **OPEN** |

# Highest-priority next work

1. **Real activation spectra.** First accessible Transformer checkpoint: cumulative calibration energy and output-sensitive rank curves at 8/16/32/64/128 for QKV/MLP/vocab sites.
2. **Competitive baseline.** Replace project row-Q4 as the success reference with actual Q4_K_M/IQ or equivalent optimized low-bit deployment at iso-context and iso-quality.
3. **Convergence study.** Multi-seed teacher/shared/smaller-model curves with tuned decay schedules; current 120-step teacher is not a convergence ceiling.
4. **Complete generator replay.** Run the committed five-seed Run-5 generator end-to-end in an environment without the current execution ceiling and diff the promoted JSON.
5. **Integrated packed runtime.** Execute group-64 packed weights and grouped latent KV in one process and measure peak RSS, not structural bytes.
6. **L3.** Convert an independent 135M+ pretrained checkpoint and evaluate real perplexity/tasks/rare-token behavior.
7. **L4.** CUDA/Metal packed kernels, measured VRAM, TTFT, tokens/s, quality.
8. **20–30× research.** Only after ≥10× passes L3/L4; then explore more aggressive structural sharing/projection/residual tiers.

# Current claim boundary

> **LARC now has controlled five-seed L2C evidence for approximately 10.86–11.30× lower modeled inference-tensor memory than the project's simple row-Q4 baseline across context 64–8192, with mean perplexity ratio 1.047 against that same baseline. Absolute mean perplexity remains 1.377× the FP32 teacher. This is synthetic character-LM evidence, not Q4_K_M parity, not measured RAM/VRAM, and not evidence for arbitrary pretrained GGUF models.**
