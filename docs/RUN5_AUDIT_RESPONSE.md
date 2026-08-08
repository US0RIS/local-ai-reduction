# Run 5 — reconciled response to the Run 4 audit

Run 5 was developed while `main` independently advanced the packed-runtime track. The work was therefore reconciled onto the newer mainline rather than merged over it.

The audit produced three distinct experimental outcomes:

1. **weight diagnosis/recovery:** depth-wise dither is weak evidence; finer scale locality plus teacher-layer function prefit and hard QAT is materially better;
2. **alternate grouped-KV track:** token-grouped FP16 metadata can restore >10× long-context arithmetic but is not the best quality/runtime path;
3. **preferred bridge:** the improved Run-5 weights combined with the already native-validated Q2/E4M3 attention codec give the strongest controlled candidate.

## Audit dispositions

### Full-stack pairing

The older Run-4 `2.45371` diagnostic was Q4 weights with ordinary/full KV. It did not include latent-Q2 KV. Run 5 therefore evaluates complete candidate stacks directly; no historical KV delta is added arithmetically.

### Correlated-error hypothesis

A six-realization stochastic-Q4 counterfactual compared one quantization-error realization reused at all 16 depths with independently resampled realizations at each depth. Mean decorrelation benefit was only **+0.0343 ± 0.0482 nats/char**, with two realizations worsening.

This does not support depth correlation as the primary cause of Q4 damage.

A stronger diagnostic is the weight distribution. Across four additional training seeds:

- independent teacher rows: mean absmax/RMS ~**2.13**, raw row-Q4 matrix NMSE ~**0.73%**;
- recovered shared block: absmax/RMS **3.10–3.24**, raw row-Q4 NMSE **1.56–1.73%**.

The shared block itself is much harder for one-scale-per-row Q4.

Artifact: `benchmarks/run5_weight_diagnostics.json`.

### Weight fix: group-64 Q4

The selected codec retains signed `[-8,7]` nibbles but stores one FP16 scale for each contiguous <=64 weights. Shared-model modeled payload rises modestly from ~77.3 KB row-Q4 to **79,828 B**, while reducing error on difficult shared rows.

A native direct-packed primitive was added:

- `Q4GroupRows` / `q4_grouped_gemv`;
- test matrix 7×130, group size64, including a partial third group;
- max abs error vs separately decoded arithmetic: **3.34e-6**;
- packed storage: **497 B**, exactly equal to the formula.

Artifact: `benchmarks/run5_native_q4_group64.json`.

### Conversion fix: function-space collapse + hard QAT

Parameter averaging alone is a poor 16→1 initialization. The strongest controlled procedure found is:

1. train the conventional 16-independent-block teacher;
2. initialize the shared block from the layer-parameter mean;
3. **80-step teacher-layer function prefit** over the union of all 16 layer input/output transformations;
4. project all matrices to group-64 Q4;
5. **200-step hard-projected QAT LM recovery** at LR `1.5e-3`.

Depth adapters and teacher-logit distillation helped selected seeds but did not improve the tested five-seed mean enough to promote. Plain QAT was less robust than function-prefit + QAT.

### Seed variance

Run 5 uses five teacher/training seeds: `3,7,11,19,23`. Single-seed effects are treated as diagnostics, not headline evidence.

### Metadata grouping

The audit correctly identified metadata as a high-leverage KV lever. Run 5 tested scalar FP16 min/scale shared across token groups. A 3-token K/V group is a workable rate-distortion point:

- modeled total reduction **11.297×** at context64;
- **10.857×** at 8K;
- five-seed mean PPL **1.047×** the project row-Q4 reference.

Artifacts: `run5_memory_context.json`, `run5_fullstack_multiseed.json`.

This remains an alternate path because it lacks a native grouped-metadata attention primitive and was slightly worse in five-seed quality than the E4M3 bridge below.

### Context-dependent scratch

The grouped reference path no longer holds scratch constant. Its accounting uses

`workspace(T) = 3584 + 80T bytes`

for the controlled geometry. The preferred E4M3 path instead uses the smaller already-defined direct-packed attention scratch contract from upstream Run 4.

### Both K/V metrics use the same ridge convention

For decoded Q4 basis `B`, both key and value paths use

`(B B^T + lambda I)^-1`,

with `lambda = 1e-5 * mean(diag(B B^T))` per head.

### Teacher-320

A naive 320-step continuation at the original constant LR degraded rather than defining a useful ceiling. It is not promoted as convergence evidence. Tuned/decayed multi-seed learning curves remain required.

### Tiny-model geometry vs SmolLM2

The controlled model uses rank16/head-dim32 = 50%; SmolLM2 structural work uses rank16/head-dim64 = 25% and GQA. The controlled tiny-model KV ratio therefore does not upper-bound SmolLM2 structural arithmetic. SmolLM2 quality remains unmeasured.

## Preferred Run-5 bridge: improved weights + E4M3 packed-attention codec

The five trained function-prefit/group64-QAT models were reevaluated using the **same latent-codec mathematics as the native packed Run-4 attention path**:

- deterministic rank16 bases from a disjoint seed-555 calibration stream;
- Q4 K/V bases;
- both FP16 inverse-Gram metrics;
- Q2 latent coefficients;
- one E4M3-FN min and scale byte per latent vector.

Evaluation is context64 on the seed-999 100,032-character stream for every training seed.

Against the project's canonical row-Q4 teacher reference:

| seed | baseline NLL | LARC NLL | delta nats/char | PPL ratio |
|---:|---:|---:|---:|---:|
| 3 | 1.91050 | 2.08693 | +0.17643 | 1.1930× |
| 7 | 2.27746 | 2.13375 | -0.14371 | 0.8661× |
| 11 | 2.14200 | 2.08938 | -0.05262 | 0.9487× |
| 19 | 2.15784 | 2.03713 | -0.12071 | 0.8863× |
| 23 | 2.02544 | 2.17927 | +0.15383 | 1.1663× |

Five-seed statistics:

- mean delta: **+0.00264 nats/char**;
- sample std: **0.15228**;
- mean perplexity ratio: **1.01208×**;
- PPL-ratio sample std: **0.15623**;
- mean PPL ratio vs FP32 teacher: **1.33287×**.

Artifact: `benchmarks/run5_e4m3_multiseed.json`; generator: `tools/run5_e4m3_multiseed.py`.

### Preferred packed byte model

Charging the larger group-64 weight payload while using the upstream direct-packed Q2/E4M3 cache/scratch contract gives:

| context | modeled total reduction |
|---:|---:|
| 64 | **11.825×** |
| 256 | **11.123×** |
| 512 | **10.856×** |
| 1K | **10.682×** |
| 2K | **10.582×** |
| 4K | **10.527×** |
| 8K | **10.499×** |

Artifact: `benchmarks/run5_packed_context_sweep.json`.

Quality is validated only at context64.

## Native evidence reconciliation

Separate native L1 primitives now exist for both major components of the preferred candidate:

- group-64 packed Q4 GEMV: `runtime/larc_q4.cpp`;
- Q2/E4M3 direct-packed latent attention: `runtime/larc_q2_attention.cpp`.

The five-seed quality path uses mathematical/dequantized execution equivalent to these storage semantics, but **the two primitives are not yet integrated into one native full-model inference loop**. Therefore no measured RSS/VRAM or integrated throughput claim is made.

## Current conclusion

The audit recommendations changed the engineering direction materially:

- dither was deprioritized after direct mechanism testing;
- finer Q4 scale locality was validated and implemented natively;
- five seeds exposed conversion variance;
- function-space prefit + hard QAT substantially improved structural conversion;
- metadata grouping was useful, but E4M3 per-vector metadata ultimately gave better five-seed quality while retaining an existing native packed attention primitive.

The next controlled milestone is now precise: **wire group-64 packed weights and packed Q2/E4M3 attention into one inference process, run the multi-seed controlled model through that native path, and measure RSS/throughput.** After that, the decisive transfer tests are a real competitive Q4_K_M/IQ baseline, real activation spectra, and an external pretrained model.
