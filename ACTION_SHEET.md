# LARC Action Sheet

Canonical technical record for **LARC — Local Adaptive Representation & Compute**. Raw evidence is under `benchmarks/`; current artifact authority is `benchmarks/ARTIFACT_MANIFEST.json`; specification is `docs/SPEC.md`.

## Objective

Reduce local-LLM **peak resident inference memory** by roughly **10–30× versus a named Q4-class baseline at the same context**, while retaining useful capability. File-size reduction alone does not count.

Report separately: serialized bytes, unique weight bytes, KV bytes, scratch, modeled tensor residency, measured RSS/device peak, quality delta (nats/token + perplexity ratio), throughput, baseline, context, and evidence level.

Evidence levels: L0 format; L1 operator/runtime; L2 controlled trained model; L2C post-training controlled conversion; L3 independent pretrained LLM; L4 measured target hardware.

---

# Run 1 — representation feasibility

## HRVQ64

Nominal 1/2/3-stage rates excluding codebook transmission were 0.1875/0.3125/0.4375 bpw. Three-stage synthetic output NMSE remained ~0.49–0.67.

Audit interpretation: at 0.4375 bpw the Gaussian squared-error rate-distortion bound is ~0.545 NMSE, so the key conclusion is information-theoretic: unstructured sub-0.5-bpw weight coding cannot preserve high fidelity. Codebook amortization is mandatory; one 256×64 FP16 codebook is 32,768 B/stage. HRVQ is retained only as a residual-page candidate.

## Activation-subspace projection

Synthetic five-operator test under deliberately concentrated activation covariance entered the target byte regime (e.g. rank-10 Q4 ~23.1× vs project row-Q4). This established a conditional mechanism, not real-LLM activation geometry.

Run 3 later changed projection fitting to quantize the basis first and solve activation-weighted least squares against the basis actually executed.

---

# Run 2 — complete-memory architecture

Introduced:

- logical graph vs physical bundles,
- recursive/shared Transformer blocks,
- projection factors,
- latent Q2 KV,
- packed Q4 CPU execution,
- Triton Q4 reference source,
- v0.2 paged mmap container,
- controlled recurrent/post-training tests,
- SmolLM2 harness.

Surviving results:

- v0.2 paged-container round trip/alignment/CRC/shared-page accounting is valid (L0).
- exact aliasing of a model whose logical definition genuinely reuses one block is lossless; 16 literal copies vs one physical object produced identical logits.
- direct packed-Q4 execution is real; dense `W=AB` need not be reconstructed.

Revoked/superseded Run-2 claims include the unreproducible `0.046151` native-operator NMSE, the mismatched post-training NLL comparison, mixed FP32/Q4 basis accounting, and mixed FP32/FP16 Q4 scale ABIs. Historical files remain only for audit history.

Clarification: the `196,608 B` in `run2_recurrent_conformance.json` is explicitly **bounded scratch** (`4*64*128*6`), not KV. That experiment omitted KV from its simple peak model.

---

# Run 3 — first audit correction

Run 3 standardized Q4, corrected projection fitting, added an inverse-Gram key metric, expanded evaluation to 100,032 characters, and created artifact-status metadata.

Important Run-3 work that remains valid:

## Canonical Q4_ROW

- q range `[-8,7]`,
- code `q+8`, low nibble first,
- FP16 row scale,
- scale `max(max(row,0)/7, max(-row,0)/8, eps)`,
- golden Python/C++ boundary vector exercises both signed endpoints.

## Projection fit

Fit against stored/dequantized basis `B_hat`:

`min_A ||W X - A(B_hat X)||_F^2`.

Rank-10 Q4 / 98%-activation-energy synthetic output NMSE improved from ~0.0260 to ~0.02483.

## Artifact finding

Recompiling the Run-2 native source gave ~0.284 NMSE rather than archived 0.046151. This established that benchmark-source provenance needed systemic enforcement, not just one corrected number.

## Run-3 claims later revoked by Run 4

The Run-3 post-training headline (10.6628× modeled tensors with ppl×1.1456) still evaluated FP32 model weights while charging Q4 weight memory. This is not representation-consistent and is revoked.

The Run-3 equal-compute control is also revoked as convergence evidence after Run 4 put all arms on one evaluation stream and discovered that the chosen continuation schedule itself degraded the teacher.

---

# Run 4 — representation-consistent audit hardening

## Audit finding 1 — V pseudo-inverse: ACCEPTED AND FIXED

Q4 basis quantization destroys exact row orthonormality for V as well as K.

Run 4 stores both:

`G_K^-1 = (B_K_hat B_K_hat^T + lambda I)^-1`

`G_V^-1 = (B_V_hat B_V_hat^T + lambda I)^-1`.

Scores use the K metric; value reconstruction uses:

`v_out = v_lat_aggregate G_V^-1 B_V_hat`.

For H=4, rank=16, head_dim=32, two Q4 bases including FP16 row scales plus both FP16 16×16 metrics cost **6,400 B**.

Files: `larc/latent_kv.py`, `tests/test_latent_kv_metrics.py`.

## Audit finding 2/3 — equal-compute control: OLD CONTROL REVOKED

All arms were evaluated on the same fresh 100,032-character stream:

- 120-step independent teacher: **1.67675 NLL**,
- same teacher continued +200 at recovery LR: **2.83686**,
- converted/recovered shared model: **2.16954**,
- recurrent-from-scratch 320-step model: **3.28752**.

The teacher itself gets much worse under the continuation schedule. Therefore the old “conversion beats scratch at equal steps” result mainly probes an unstable/poor early-training schedule and is not convergence evidence.

Artifact: `benchmarks/run4_equal_compute_schedule_failure.json`.

Open requirement: compare converged curves with tuned/matched schedules and include teacher-at-budget ceiling.

## Audit finding 4 — Q4 quality consistency: ACCEPTED; INITIAL RESULT FAILED

Teacher and student were quantized to the canonical Q4 representation actually charged in memory.

Before Q4-aware recovery, same 100,032-character stream:

- Q4 teacher: **1.88548 NLL**,
- Q4 shared student: **2.62682**,
- shared + corrected latent Q2: **2.66766**,
- total perplexity ratio: **2.186×**.

This confirmed the audit's concern: quantization error in one physical block reused 16 times is correlated through depth and can be much more damaging than quantizing 16 independent blocks.

Artifact: `benchmarks/run4_q4_consistency_failure.json`.

## Projected-Q4 recovery

Added representation-aware recovery:

1. start from structurally recovered shared model,
2. project every matrix to canonical Q4 and 1-D tensor to FP16,
3. optimize at LR `3e-4`,
4. after every optimizer step immediately project back onto the storage grid,
5. choose among 25-step checkpoints on a **disjoint selection stream seed 444**.

Best selection checkpoint: projected-Q4 step 150. Final independent stream NLL before KV compression: **1.94078**.

Artifact: `benchmarks/run4_q4_projected_recovery.json`.

Additional recovery compute must always be disclosed: 120 teacher steps + 200 structural-recovery steps + up to 200 projected-Q4 steps.

## Evaluation-stream hygiene

During Run-4 reproducibility work, another leak was found: early latent-basis fitting used final-evaluation contexts. That result was invalidated.

Current four streams are independently generated:

- training: seed **3**,
- Q4-recovery checkpoint selection: **444**,
- latent-basis calibration: **555**,
- final evaluation: **333**.

Final evaluation length: **100,032 characters**.

Basis fitting is now deterministic uncentered eigendecomposition of `X^T X` rather than randomized PCA.

## FP8 metadata profile

For rank-16 row-Q2 latent K/V, FP16 min+scale metadata consumed as many bytes as the 2-bit coefficients. Run 4 added E4M3-FN metadata:

- 4 coefficient bytes/vector,
- 1-byte min,
- 1-byte scale,
- 6 B per K or V vector,
- 12 B for K+V vs 128 B FP16 K+V at head_dim 32,
- raw KV ratio **10.6667×**.

Clean context-64 quality result:

| path | NLL |
|---|---:|
| independent Q4 teacher | **1.88548** |
| Q4-recovered shared, normal KV | **1.94078** |
| Q4-recovered + latent Q2 + FP8 metadata + both metrics | **1.97525** |

Decomposition:

- structural conversion/Q4 recovery: **+0.05529 nats/char**,
- KV path: **+0.03447 nats/char**,
- total: **+0.08977 nats/char**,
- total perplexity ratio: **1.09392×**.

Artifact: `benchmarks/run4_fp8meta_l2c.json`.

This is the strongest current controlled quality result.

## Audit finding 7 — native operator floor: QUANTIFIED

For `W=AB+sigma*epsilon`, the rank-32 irreducible residual fraction is `sigma^2/(1/576+sigma^2)`.

At sigma=0.02:

- theoretical structural floor: **0.18726**,
- measured output NMSE vs exact W: **0.26843**,
- vs direct Q4: **0.28779**.

At sigma=0.002:

- structural floor: **0.00230**,
- output NMSE vs exact W: **0.03330**,
- vs direct Q4: **0.05663**.

The low-noise case is the cleaner factor-quantization diagnostic.

Artifacts: `tests/native_q4_fidelity.cpp`, `benchmarks/run4_native_q4_fidelity.json`.

## Audit finding 5 — artifact provenance: SYSTEMIC FIX

`benchmarks/ARTIFACT_MANIFEST.json` marks every important artifact as current/historical/superseded/revoked.

Quick current structural artifacts regenerate using:

`python tools/check_quick_benchmark_artifacts.py`.

`.github/workflows/reproducibility.yml` runs Python tests, deterministic artifact checks, native Q4 tests, and packed-Q2 attention tests when GitHub runners are available.

Heavy training reproducer:

`tools/run4_l2c_repro.py`

with protocol in `docs/RUN4_REPRO.md`.

Historical JSON is not silently treated as current evidence merely because it remains in the repository.

## Audit finding 6 — Q4 scale: ALREADY CORRECT, NOW NORMATIVE

The Run-3 implementation already used the no-clipping asymmetric signed rule:

`scale=max(max_positive/7, abs(min_negative)/8)`.

Run 4 retains it and makes it normative in `docs/SPEC.md`; golden tests exercise both -8 and +7 codes.

## Audit findings 8/9 — KV metadata and basis scales: ACCEPTED

Basis byte accounting now includes every Q4 row scale. Both K and V metrics are charged.

E4M3-FN min/scale cuts row-Q2 metadata from 8 B/token (K+V) to 4 B/token, increasing the raw rank-16/head-dim-32 KV ratio from 8× to 10.667× without reducing rank.

## Context sweep — REINSTATED AS A HARD GATE

The older Python-reference scratch model materializes a `T×r` latent decode buffer. With FP8 metadata it is:

- context 64: **11.74×** modeled total,
- 1K: **10.05×**,
- 2K: **9.91×**,
- 8K: **9.79×**.

Therefore metadata compression alone did not solve long-context total memory.

### Direct packed latent-Q2 attention

Implemented `runtime/larc_q2_attention.{h,cpp}` plus `q4_transposed_gemv`.

The kernel directly consumes:

- packed Q2 K/V,
- E4M3-FN metadata,
- Q4 K/V bases,
- FP16 inverse-Gram matrices.

It never creates historical FP32 `T×r` K/V arrays. Scratch is `T + 4r` FP32 values for one head and can be reused across heads.

L1 test at T=2048, rank16, head_dim32:

- max abs error vs separately decoded reference: **2.50e-9**,
- packed cache per head: **24,576 B**,
- direct scratch per head: **8,448 B**,
- full decoded FP32 latent K+V history: **262,144 B**.

Artifact: `benchmarks/run4_native_q2_attention.json`.

### Packed-runtime structural context model

Using fair autoregressive scratch accounting (baseline does not get charged LARC's removed `T×r` buffer):

| context | modeled total reduction |
|---:|---:|
| 64 | **12.04×** |
| 256 | **11.22×** |
| 512 | **10.91×** |
| 1K | **10.71×** |
| 2K | **10.60×** |
| 4K | **10.53×** |
| 8K | **10.50×** |

Artifact: `benchmarks/run4_packed_attention_context_sweep.json`.

**Critical boundary:** quality has only been validated at context 64. The 2K/8K values are structural/runtime byte models, not long-context quality results or measured RAM/VRAM.

## SmolLM2-shaped structural plan after corrections

The planner already uses SmolLM2 GQA (`kv_heads=3`, head_dim=64). After charging Q4 basis row scales and **both** inverse-Gram metrics, rank-16 KIVI-shaped structural KV ratios are:

- **18.245× at 2K**,
- **19.309× at 8K**.

For the nominal `10x` weight profile, modeled total is ~**13.87× at 2K** and **16.17× at 8K**.

Artifact: `benchmarks/run4_smollm2_structural_rank16.json`.

**Quality is completely unvalidated for these SmolLM2 ranks/profile assumptions.** This remains structural arithmetic only.

---

# Current claim boundary after Run 4

The strongest defensible statement is:

> On a controlled synthetic character-LM post-training conversion, with quality paths executing the same canonical Q4 weight representation charged by the memory model, with disjoint training/checkpoint-selection/basis-calibration/final-evaluation streams, a rank-16 latent-Q2/E4M3 KV representation produced **+0.08977 nats/char** total degradation (perplexity ×**1.09392**) at **context 64**. Combining that L2C codec/model result with the separately L1-validated direct packed-attention scratch contract gives a **12.04× modeled same-context inference-tensor reduction**.

Do **not** state that LARC has demonstrated 10–30× lower measured RAM/VRAM for real pretrained GGUF models.

## Original-goal status

| Goal | Current status |
|---|---|
| new runtime-native model format | **implemented research format (L0)** |
| direct compressed Q4 execution | **implemented/tested CPU (L1)** |
| direct packed latent-Q2 attention | **implemented/tested CPU (L1)** |
| >10× modeled controlled total memory | **yes at context64; packed structural model remains >10× through 8K** |
| reasonable controlled quality | **ppl ×1.09392 at context64 L2C** |
| representation-consistent Q4 quality | **yes in Run4 controlled test** |
| long-context quality | **OPEN** |
| converged equal-compute comparison | **OPEN; old control revoked** |
| real activation spectra | **OPEN** |
| independent pretrained 135M+ conversion | **OPEN (L3)** |
| actual process RSS / GPU/Metal VRAM ≥10× | **OPEN (L4)** |
| 20–30× real-model quality | **OPEN** |
| competitive iso-byte comparison vs IQ/AQLM/QuIP#/smaller dense | **OPEN** |

## Next controlling milestones

1. **Real activation geometry:** measure cumulative spectra at each SmolLM2 attention/MLP site, especially ranks 8/16/32/64.
2. **Long-context controlled quality:** extend positional/context training and validate 256→8K quality using the direct packed codec semantics.
3. **Convergence control:** train teacher/shared-from-scratch/conversion arms with stable tuned schedules to convergence, multiple seeds, and teacher-at-budget ceiling.
4. **Full packed runtime integration:** execute Q4 weights and packed latent KV together in one inference program, then measure real process RSS.
5. **L3:** external pretrained checkpoint, real perplexity/tasks/generation, including rare-token behavior and vocabulary-factor ablations.
6. **L4:** CUDA/Metal/CPU measured peak memory and optimized throughput against named baselines.
7. **Competitive baselines:** Q4_K_M/IQ quants, AQLM/QuIP# where runnable, and smaller dense models at equal bytes.
