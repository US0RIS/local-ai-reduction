# LARC Action Sheet

This file is the persistent technical project log. It should be updated on every substantive project run.

## Project objective

Create a new local-AI model storage/execution standard that makes capable models practical on materially weaker devices than ordinary GGUF deployment. Initial stretch target: **10–30× smaller stored/resident representation than a Q4-class GGUF baseline**, without pretending that raw compression ratio alone constitutes success.

### Success criteria

A result only counts as a breakthrough if it simultaneously improves:

1. stored model bytes,
2. resident RAM/VRAM,
3. retained model capability,
4. practical inference speed or at least acceptable slowdown,
5. conversion practicality (prefer streaming source shards rather than requiring a second full local copy).

---

# Run 1 — 2026-08-08

## Status

**Format architecture established; two structural codec families implemented; first synthetic experiments completed. Raw 10–30× storage regime is reachable, but intelligence retention at that ratio is not yet demonstrated.**

## Repository starting state

The repository contained only `# local-ai-reduction` in `README.md`. No implementation or prior benchmark artifacts existed.

## Standard decision

Working standard name: **LARC — Local Adaptive Representation & Compute** (`.larc`).

Decision: do **not** build a new container around the same dense Q2/Q4 tensors. The standard's logical operator representation is:

`activation-subspace core + progressive residual refinements + sensitive-tensor fallbacks`.

The format is designed as an execution representation. GGUF/SafeTensors remain import sources.

## Research review

Reviewed prior art relevant to the design:

- AQLM: additive multi-codebook quantization; strong results around ~2 bits/parameter.
- QuIP#: randomized Hadamard incoherence + lattice vector quantization.
- SqueezeLLM: dense/sparse sensitivity-aware quantization.
- SVD-LLM / ASVD: activation-aware and truncation-aware low-rank decomposition.
- CALDERA: activation-weighted low-rank + low-precision decomposition.
- BitStack: progressive residual blocks for variable-memory model sizing.
- bitnet.cpp: evidence that storage representation and kernels must be co-designed.
- llama.cpp GGUF quantization tables: practical baseline.

**Implication:** ordinary weight-only quantization is already highly optimized around 2 bits/weight. A 10–30× improvement over Q4 implies roughly 0.15–0.45 effective bits/original-weight if achieved solely through weight payload compression, which is too aggressive to expect from scalar/vector quantization alone. LARC therefore has to exploit functional/activation structure and runtime selectivity.

## Implemented prototype components

### 1. Minimal LARC container

`larc/container.py`

- 8-byte magic/version,
- uint64 manifest length,
- JSON research manifest,
- concatenated binary chunks,
- manifest parser test.

This is intentionally temporary; a binary manifest and aligned page table should wait until codec requirements stabilize.

### 2. HRVQ64 progressive residual codec

`larc/hrvq.py`

Current layout:

- vector width = 64 weights,
- one FP16 RMS scale per 256 weights,
- 256-entry learned residual codebooks,
- one uint8 code index per 64 weights per stage,
- additive refinement stages.

Nominal payload rates:

| Stages | Nominal bpw | Ideal multiple vs 4.5 bpw Q4-class payload |
|---:|---:|---:|
| 1 | 0.1875 | 24.0× |
| 2 | 0.3125 | 14.4× |
| 3 | 0.4375 | 10.29× |

For a hypothetical 135M-weight model with one globally amortized codebook set, estimated weight payloads are:

| Stages | Estimated payload + codebooks | Ratio vs 105 MB SmolLM2-135M Q4_K_M file* |
|---:|---:|---:|
| 1 | 3.20 MB | 32.85× |
| 2 | 5.34 MB | 19.67× |
| 3 | 7.48 MB | 14.04× |

*Not apples-to-apples yet: the LARC estimate excludes tokenizer/manifest and non-matrix treatment, while the 105 MB value is the complete published GGUF. This table is a target-scale estimate, not a demonstrated model conversion.

### HRVQ synthetic quality result

Benchmark: 768×768 matrices; codebook overhead included in effective bpw for this small test. The raw codec had high distortion:

- Gaussian, 3 stages: weight NMSE ~0.672; output NMSE ~0.671.
- Heavy-tail, 3 stages: weight NMSE ~0.615; output NMSE ~0.607.
- Low-rank-plus-noise, 3 stages: weight NMSE ~0.479; output NMSE ~0.490.

**Decision:** HRVQ is not a viable base representation at these bitrates. Keep it as a residual/refinement codec where the remaining signal is lower-importance and activation-weighted.

### 3. Activation-subspace projection bundles

`larc/projection.py`, `larc/q4.py`, `larc/q8.py`

For multiple operators `W_i` consuming the same input activation space, learn a calibration basis `U_k` and store `A_i = W_i U_k`.

Runtime computes `z = U_k^T x` once, then `y_i = A_i z` for every operator in the bundle.

This exploits a property GGUF cannot express: multiple logical tensors share one learned input representation.

### Projection-bundle synthetic benchmark

Setup:

- five 384×384 linear operators,
- shared calibration activation covariance,
- top subspace holds 90%, 95%, or 98% of activation variance,
- held-out activations sampled from the same covariance,
- baseline = actual bytes from the prototype row-wise Q4 encoder.

Best storage-target cases:

| Core rank | Factor precision | Compression vs row-Q4 | Held-out output NMSE at 95% core energy | At 98% core energy |
|---:|---:|---:|---:|---:|
| 10 / 384 = 2.60% | Q4 | **23.10×** | 0.0557 | 0.0260 |
| 10 / 384 = 2.60% | Q8 | **13.47×** | 0.0499 | 0.0200 |
| 19 / 384 = 4.95% | Q4 | **13.47×** | 0.0578 | 0.0280 |

Interpretation:

- The **10–30× storage band is mechanically reachable** for a projection core when activations are strongly concentrated.
- Q8 factors reduce quantization error while still landing above 10× at very low retained rank.
- The dominant unanswered question is whether real LLM layer activations have enough compressible subspace, consistently enough across prompts/domains, to preserve actual language-model capability.
- Synthetic covariance results must not be represented as model benchmark results.

## Test status

`pytest -q` → **3 passed**.

Tests cover HRVQ shape/finite output, container manifest write/read, and Q4/projection-bundle execution.

## Known limitations / blockers

1. **No real-model quality benchmark yet.** The Run 1 environment could inspect Hugging Face metadata but could not retrieve the 269 MB SmolLM2-135M SafeTensors object into the execution sandbox. The published source size is 269 MB and an available Q4_K_M GGUF is ~105 MB, so SmolLM2-135M remains the first intended real target.
2. Projection bundles assume calibration activations generalize. Cross-domain held-out calibration tests are required.
3. Current factor quantizers are simple row-wise Q4/Q8 reference implementations, not optimized kernels.
4. HRVQ codebooks are learned with MiniBatchKMeans and are not activation-aware yet.
5. No sparse rescue path, Hadamard transform, or error-feedback optimization is implemented.
6. No direct compressed-domain C/C++ kernel exists yet.
7. Container chunks are not aligned/paged/checksummed in v0.1.

## Next experiments — ordered

### P0 — Real model validation

- Build a streaming SafeTensors reader/converter that processes one source tensor/shard at a time.
- Target SmolLM2-135M first, then 360M, before moving to 1B+.
- Capture calibration activations for each bundle candidate.
- Measure activation covariance spectra per layer and tensor family.
- Determine whether 2–5% retained dimensions can capture enough task-relevant activation energy.
- Evaluate perplexity and generation quality at 10×, 15×, 20×, and 30× storage targets.

### P0 — Residual rescue

Optimize residuals against `||(W - W_core - R) X||_F^2`, not raw weight Frobenius error.

Candidate residual stack:

1. sparse high-impact directions/entries,
2. Hadamard/incoherence transform,
3. HRVQ64 additive pages,
4. optional low-rank error correction.

Pages should be ordered by marginal validation gain per byte.

### P1 — Bundle discovery

Automatically group operators with common input spaces. Attention Q/K/V and MLP gate/up are initial candidates.

### P1 — Runtime kernel

Implement CPU reference kernels for `A(U^T x)` directly from Q4/Q8 factors, without materializing dense weights. Measure memory bandwidth, extra arithmetic, cache behavior, and break-even rank versus Q4 GGUF.

### P1 — Format hardening

After real-model evidence: fixed binary manifest, 4–256 KiB page alignment experiments, checksums, dependency graph, quality-tier profiles, deterministic codec IDs, and architecture/tokenizer mappings.

## Current assessment

**Most promising direction:** projection bundles plus activation-weighted progressive residual pages.

**Deprioritized as a standalone solution:** sub-0.5-bpw vector-codebook coding of entire raw weight matrices. It meets the byte target but Run 1 distortion is too high.

**Confidence:** low-to-moderate that 10× over Q4 can be made useful on at least some models/layers; low that 20–30× will retain broadly comparable intelligence without strong activation subspace concentration, calibration/fine-tuning, conditional paging, or architecture-aware transformations. The project should test this rather than assume it.
