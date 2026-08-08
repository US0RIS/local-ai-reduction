# Run 5 — response to the Run 4 external audit

This run treats the Run-4 audit as an experimental specification. The project does not preserve a headline merely because its arithmetic is internally consistent; quality must execute the same representation whose bytes are charged, multi-seed distributions replace favorable single-seed results, and every memory result names context and baseline.

## Audit findings accepted

### Full-stack quality pairing

Run 4's `2.45371` NLL was the converted model with row-Q4-dequantized weights and ordinary/full KV. It did **not** include latent-Q2 KV. Therefore the old latent-KV delta could not simply be added: it came from a superseded Run-3 path. Run 5 evaluates the weight and KV representation jointly.

### Correlated-error attribution was confounded

A six-realization stochastic-Q4 counterfactual compared one quantization error field reused at all 16 depths with independent stochastic realizations at each invocation. Mean benefit from decorrelation was only `+0.0343 ± 0.0482` nats/char and changed sign in two realizations. Correlation may contribute but is not established as the main mechanism.

A stronger diagnostic is the weight distribution itself. Across seeds 7/11/19/23, independent teacher rows had mean absmax/RMS ~2.13 and raw row-Q4 weight NMSE ~0.73%; the shared recovered block had absmax/RMS ~3.10–3.24 and row-Q4 NMSE ~1.56–1.73%. The shared block is intrinsically harder for one-scale-per-row Q4.

### Multi-seed evidence is load-bearing

Run 5 uses training seeds `3,7,11,19,23` and the same independently generated seed-999 evaluation stream, 100,032 characters per seed. Single-seed engineering decisions are treated as diagnostics only.

### Metadata grouping has large leverage

At d_head=32, rank=16, tokenwise Q2 spends half the coefficient+metadata stream on min/scale metadata. Run 5 groups one FP16 min/scale pair across both token and latent dimensions for K and V. Group size is a rate-distortion parameter; larger groups improve bytes but degrade quality.

The selected controlled candidate uses **3-token groups** because it preserves >10× modeled total-memory reduction across the entire 64–8192 context sweep while being less aggressive than group 4/8.

### Scratch scales with context

Reference workspace is no longer held at 8,704 B. For the tiny controlled model the Run-5 reference formula is:

`workspace(T) = (D + D + 3D + FF + H*T + rank*T) * 4 = 3584 + 80T bytes`.

This is still a modeled reference workspace, not a profiler trace.

### Both inverse-Gram corrections use the same ridge rule

For each Q4-dequantized K/V basis `B`, Run 5 uses

`G^-1 = (B B^T + lambda I)^-1`,

with `lambda = 1e-5 * mean(diag(B B^T))` per head. The K score and V pseudoinverse paths both use this rule.

### Teacher-320 is not a convergence ceiling

A constant-LR 320-step teacher run degraded rather than improving under the current schedule. Therefore `teacher-320 at the same LR` is not promoted as a convergence control. The correct remaining experiment is a convergence study with tuned/decayed schedules and learning curves.

### The tiny-model long-context asymptote does not bound SmolLM2

The tiny controlled model uses rank 16 of head_dim 32 (50%). SmolLM2's structural planner uses rank 16 of head_dim 64 (25%) and GQA. Consequently the controlled model has less favorable KV geometry. No claim is made that the tiny-model KV ratio upper-bounds SmolLM2; SmolLM2's 18–19× KV arithmetic remains structural-only until real quality is measured.

## Weight-side experiments

### Dither diagnostic

Artifact: `benchmarks/run5_weight_diagnostics.json`.

Result: depth-wise error decorrelation is weak and noisy. It does not justify adding a stochastic/depth-rotated codec as the primary fix.

### Finer scale grouping

The selected weight representation uses one canonical signed Q4 scale for each **64-weight sub-row group** rather than each full row. Shared-model payload rises modestly from ~77.3 KB under row-Q4 to **79,828 B**, but quantization error falls materially on difficult shared blocks.

This is consistent with the low-noise factor benchmark: two row-Q4 factor stages incur approximately the expected absmax-driven quantization loss. Finer scale locality is a better-targeted lever than simply increasing both factors to Q8.

### Depth adapters

Small per-depth low-rank residual adapters helped some seeds but worsened the five-seed mean under the tested recovery schedule. They remain a format feature/candidate, not the current best controlled configuration.

### Teacher-output distillation

Teacher-logit distillation modestly improved some FP32 recovery cases but did not improve the actual quantized full stack enough to promote.

### Hard-projected quantization-aware recovery

After each optimizer step, all matrix weights are projected back to the exact group-64 Q4 representation. This materially improves robustness on difficult seeds and ensures the recovery objective optimizes the representation actually stored.

### Teacher-layer function prefit

Parameter averaging alone is a poor 16→1 collapse initialization. Run 5 adds 80 steps in which the shared block is trained on the union of all teacher-layer input/output transformations: each training batch is passed through the 16 independent teacher blocks, all layer `(input, output)` pairs are collected, and the one shared block is fit by MSE across those 16 roles.

Then the model undergoes 200 steps of hard-projected group-64 QAT language-model recovery at LR `1.5e-3`.

This combination is the strongest controlled structural conversion method found so far.

## Selected Run-5 controlled candidate

- teacher: 16 independently parameterized Transformer blocks, 120 training steps;
- LARC physical graph: one shared Transformer block invoked at 16 logical depths;
- collapse initialization: parameter mean + 80-step teacher-layer function prefit;
- recovery: 200-step hard-projected group-64 QAT;
- LARC weights: signed Q4 `[-8,7]`, one FP16 scale per 64-weight sub-row group;
- KV rank: 16 of head_dim 32;
- KV coefficients: Q2;
- K metadata: one FP16 min + FP16 scale per 3-token group;
- V metadata: same 3-token grouping;
- K/V bases: Q4, one physically shared basis set across all logical depths;
- K/V basis metrics: FP16 ridge-stabilized inverse-Gram matrices;
- incomplete group: FP16 residual tail, explicitly charged and used during quality evaluation;
- evaluation: same seed-999 100,032-character stream for each training seed.

## Memory result

Baseline is **the project's simple row-Q4 teacher + FP16 KV + the same reference workspace**, not llama.cpp Q4_K_M.

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

These are structural tensor-accounting results. They are not measured RSS or VRAM.

## Full-stack quality result

Artifact: `benchmarks/run5_fullstack_multiseed.json`.

Against the **same project row-Q4 teacher baseline used for memory**:

| seed | baseline Q4 NLL | LARC full-stack NLL | delta nats/char | perplexity ratio |
|---:|---:|---:|---:|---:|
| 3 | 1.91050 | 2.12264 | +0.21214 | 1.2363× |
| 7 | 2.27746 | 2.18102 | -0.09644 | 0.9081× |
| 11 | 2.14200 | 2.10687 | -0.03513 | 0.9655× |
| 19 | 2.15784 | 2.04906 | -0.10878 | 0.8969× |
| 23 | 2.02544 | 2.23119 | +0.20575 | 1.2284× |

Five-seed statistics versus row-Q4 baseline:

- mean delta: **+0.03551 nats/char**;
- sample std: **0.16078 nats/char**;
- mean perplexity ratio: **1.04705×**;
- sample std of perplexity ratio: **0.17120**;
- range: **0.8969×–1.2363×**.

Against the FP32 teacher, however:

- mean delta: **+0.31938 nats/char**;
- mean perplexity ratio: **1.37724×**.

This distinction is mandatory. LARC is roughly at parity on average with the project's primitive row-Q4 baseline, but remains materially worse than the FP32 teacher. No equivalence to optimized Q4_K_M quality has been established.

## Provenance

`tools/run5_fullstack_protocol.py` contains the self-contained five-seed training/conversion/evaluation protocol. `tools/run5_fullstack_protocol_fp16tail.py` applies the exact FP16 incomplete-group semantics used by the promoted result.

The final quality phase was rerun for all five retained trained models after the FP16-tail correction. A complete five-seed one-process replay of the committed generator has **not** independently completed after commit because the available execution ceiling terminates the long job. `benchmarks/INDEX.json` records this explicitly rather than calling the artifact CI-reproduced.

## Current interpretation

Run 5 re-establishes the controlled **>=10× modeled tensor-memory** gate across 64–8192 context **only against the project's simple row-Q4 baseline**. It does so with five-seed mean perplexity ratio ~1.047 against that same baseline.

This is a meaningful L2C milestone because:

1. memory and quality now refer to the same weight/KV representations;
2. the baseline object is paired consistently across memory and quality;
3. context-dependent workspace and incomplete-group tail are charged;
4. five seeds replace a favorable single seed;
5. conversion quality was improved by a function-space collapse objective plus QAT rather than by hiding loss in accounting.

It is not the project's ultimate success criterion. The decisive open tests remain:

- real activation spectra on an independently pretrained Transformer;
- actual Q4_K_M/IQ/AQLM/QuIP#-class iso-byte comparison;
- external pretrained 135M+ conversion and standard task/perplexity evaluation;
- integrated packed runtime with measured RSS;
- CUDA/Metal peak memory and throughput;
- 20–30× retained-quality regime.
