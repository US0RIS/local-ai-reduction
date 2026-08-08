# LARC Action Sheet

Canonical technical record for **LARC — Local Adaptive Representation & Compute**. Historical Run 1–3 details remain in `docs/RUN3_AUDIT_CORRECTIONS.md` and the benchmark artifacts; this sheet records the current claim boundary after the second external audit.

## Objective

Reduce peak resident local-LLM inference memory by roughly **10–30× versus a named Q4-class baseline at the same context length**, while retaining useful capability. Every result must identify context, baseline, quality metric, and whether memory is measured or modeled.

Evidence levels remain L0 format, L1 operator, L2 controlled model, L2C post-training conversion, L3 external pretrained model, L4 measured hardware.

---

# Run 4 — second audit closure

## 1. Value-basis correction — IMPLEMENTED

Run 3 corrected non-orthogonality of the quantized key basis but reconstructed latent values as `v_lat @ B_hat`, which is only correct when the stored basis rows remain orthonormal.

Run 4 changes value reconstruction to the pseudoinverse form:

`v_full = v_lat (B_hat B_hat^T)^-1 B_hat`.

Both key and value Q4 bases therefore store FP16 inverse-Gram matrices. `tests/test_latent_basis_metric.py` verifies that the corrected key score and value reconstruction equal the orthogonal projector for deliberately non-orthonormal row bases.

The extra value metric and exact Q4 basis-scale bytes are now charged in memory accounting.

## 2. Q4 format boundary — ALREADY CLOSED IN RUN 3

The second audit correctly asked for a pinned scale derivation, but the current main source already has it:

- integer range `[-8,7]`;
- code = `q + 8`;
- scale = `max(max_positive/7, max_negative_magnitude/8)`;
- FP16 row scale;
- low nibble first.

`tests/test_q4_format.py` exercises both code boundaries using a row containing `-8` and `+7`. No Run-4 format change was needed.

## 3. Native factor-fidelity benchmark — REDESIGNED

The old `W=AB+0.02 epsilon` benchmark had an unavoidable rank-32 NMSE floor of about 0.187, so it was unsuitable for isolating factor quantization.

New source: `tests/native_q4_fidelity.cpp` with residual noise std `0.002`.

Measured locally:

- resident factor reduction: **12.06239×**;
- theoretical source rank-32 floor: **0.002299 NMSE**;
- projected Q4 output NMSE vs exact FP32 W: **0.033299**;
- projected Q4 output NMSE vs direct row-Q4: **0.056634**.

Artifact: `benchmarks/run4_native_q4_fidelity.json`.

**Interpretation:** once truncation is nearly removed, current Q4 factor quantization still contributes substantial operator error. Better factor quantization is required before projection factors can be treated as high-fidelity.

## 4. Equal-compute control on the 100k evaluation stream — REPRODUCTION SUPPORTS CONVERSION, PROVENANCE ISSUE FOUND

The second audit correctly noted that Run 3's equal-compute control used only 32 contexts, not the 100,032-character stream used by the L2C headline.

A Run-4 reconstruction of the documented protocol uses:

- training seed 3;
- 90% of the generated training corpus;
- evaluation stream seed 999;
- **100,032 evaluation characters**;
- teacher 120 steps;
- converted student +200 recovery steps;
- scratch recurrent model 320 steps.

Independent reconstruction produced:

- teacher FP32 NLL: **1.81268**;
- recovered converted student FP32 NLL: **2.01538**;
- scratch recurrent 320-step FP32 NLL: **2.92223**.

Thus the same-stream control still strongly favors teacher→shared warm-start conversion over scratch recurrent training at the same optimizer-step count.

However, these numbers do **not** match the archived Run-3 100k artifact (teacher 1.77359 / recovered student 1.88556). The Run-3 headline artifact has no committed canonical generator script. Therefore Run 3's exact numerical headline is now classified as **historical, missing canonical generator**, not reproducibly promoted evidence.

Current generator: `tools/run4_control_reproduction.py`.
Artifacts: `benchmarks/run4_control_reproduction.json`, `benchmarks/run4_q4_weight_quality.json`.

## 5. Weight-quality vs weight-accounting mismatch — CONFIRMED MATERIAL

The second audit correctly identified that previous quality comparisons executed FP32 weights while memory accounting assumed Q4 weights.

Run 4 evaluates the same reconstructed teacher and converted student after applying the canonical row-Q4 representation to every 2-D parameter and FP16 rounding to 1-D parameters.

On the same 100,032-character seed-999 stream:

| path | FP32 NLL | dequantized-Q4 NLL | Q4 delta |
|---|---:|---:|---:|
| 16 independent teacher blocks | 1.81268 | **2.04418** | **+0.23149 nats/char** |
| one recurrent converted block | 2.01538 | **2.45371** | **+0.43832 nats/char** |

The recurrent path suffers **+0.20683 nats/char more Q4 damage** than the teacher beyond the teacher's own Q4 degradation.

This is consistent with the audit hypothesis that quantization error in one reused block is depth-correlated and can compound through the residual stream.

**Consequence:** the former L2C claim coupling FP32 quality with Q4 weight bytes is no longer sufficient. A future headline must evaluate the weight representation actually charged in memory.

## 6. Context sweep — RESTORED AS A HARD GATE

With the corrected row/row latent-Q2 cache used by the controlled L2C model, Q4 bases, both inverse-Gram matrices, and exact basis-scale bytes, modeled total-memory reduction is strongly context-dependent:

| context | modeled total reduction |
|---:|---:|
| 64 | **10.5245×** |
| 128 | **9.7843×** |
| 256 | **9.1247×** |
| 512 | **8.6466×** |
| 1K | **8.3495×** |
| 2K | **8.1821×** |
| 4K | **8.0930×** |
| 8K | **8.0470×** |

Artifact: `benchmarks/run4_context_sweep.json`.

Therefore the old `10.66× total memory` statement is valid only near the 64-token controlled context and is not a general long-context result. At practical contexts this particular codec asymptotically approaches roughly **8×** total reduction.

Every future total-memory claim MUST include context length in the headline.

## 7. SmolLM2 structural planner — ACCOUNTING CORRECTED AGAIN

Run 4 adds:

- value inverse-Gram storage;
- Q4 basis row-scale storage omitted by the older planner.

For rank-16 KIVI-style latent KV:

- KV reduction at 2K: **18.245×**;
- KV reduction at 8K: **19.309×**.

For the nominal 10x weight profile, modeled total reduction becomes:

- 2K: **13.873×**;
- 8K: **16.173×**.

These remain structural arithmetic only. No SmolLM2 quality result exists.

Artifact: `benchmarks/run4_kivi_memory_plan_rank16.json`.

## 8. Old 6.63× loose end — RESOLVED

The 196,608 B term in `run2_recurrent_conformance.json` was not a KV cache. The source defines it directly as scratch:

`4 * 64 * 128 * 6 = 196,608 B`.

That experiment did not include KV memory at all. The old 6.63× ratio is reconstructible as weight bytes plus this manually modeled scratch term.

## 9. Benchmark artifact provenance — SYSTEMIC GUARD ADDED

The stale Run-2 operator artifact and the missing generator for the Run-3 100k model artifact are treated as a systemic reproducibility issue.

Added:

- `benchmarks/INDEX.json`: classifies artifacts as current, historical, superseded, or missing-generator;
- `tools/check_benchmark_provenance.py`: current artifacts must have committed generators;
- `.github/workflows/benchmark-provenance.yml`: PR check for provenance plus deterministic context-sweep regeneration;
- Run-4 current artifacts all have committed generators.

Historical artifacts are retained for audit history rather than silently overwritten.

The hosted Actions environment previously failed before allocating steps, so the workflow's existence does not constitute a successful CI execution until GitHub actually runs it.

## 10. Q4 scale audit note

The audit suggestion to use llama.cpp Q4_0's signed-extremum convention is reasonable as an alternative codec, but LARC's candidate codec is intentionally pinned to the range-aware rule above. It uses both negative and positive endpoints without clipping either row extremum. Competitive comparison against Q4_0/Q4_K/IQ codecs remains open.

---

# Current claim boundary after Run 4

The former Run-3 headline is **downgraded** because:

1. value-basis pseudoinverse bytes were missing;
2. total reduction exceeds 10× only at the 64-token controlled context for the current row/row KV codec;
3. Q4 weight quality was not measured and is materially worse for the recurrent student;
4. the exact Run-3 100k artifact lacks a committed generator and an independent reconstruction does not reproduce its NLL values.

What remains defensible:

- L0 paged format implementation;
- lossless aliasing when a logical model truly reuses identical parameters;
- direct packed CPU Q4 execution property;
- synthetic projection feasibility under deliberately favorable activation geometry;
- controlled evidence that teacher→shared warm-start recovery trains much better than a scratch recurrent model under the tested finite-step budget;
- corrected structural memory arithmetic;
- evidence that recurrent depth sharing makes naive row-Q4 quantization substantially more damaging.

What is **not currently passed** as a complete headline gate:

- ≥10× total memory at practical context with the exact weight+KV representation whose quality is measured;
- converged equal-compute comparison;
- measured process RSS or device VRAM;
- external pretrained LLM conversion;
- real-model activation-rank validation;
- 20–30× retained-quality result;
- competitive iso-byte comparison against optimized GGUF/IQ/AQLM/QuIP# or a smaller dense model.

## Highest-priority next experiments

1. **Fix recurrent weight quantization.** Test Q8 factors, per-block residual rescue, depth-conditioned adapters, and quantization-aware recovery so reused-block error does not compound coherently.
2. **Improve KV metadata efficiency.** The current controlled row/row Q2 codec spends half of its per-token bytes on min/scale metadata. Test grouped value metadata and error-vs-attention-entropy before lowering latent rank further.
3. **Convergence study.** Teacher-120, teacher-320, converted/recovered, and scratch recurrent arms with matched/tuned schedules, multiple seeds, learning curves, and the same 100k evaluation stream.
4. **Canonical artifact regeneration.** Re-run every promoted/current artifact from committed generators and reject any value without source provenance.
5. **Real activation spectra.** First accessible Transformer checkpoint: cumulative energy at ranks 8/16/32/64/128 by projection site.
6. **Integrated packed runtime + measured RSS.** Weight Q4/Q8 + latent KV in one inference process.
7. **L3/L4.** Independent 135M+ pretrained model, real quality suite, then CUDA/Metal memory and throughput.

## Current status

**The project no longer claims that the controlled experiment has already met the complete 10× memory+quality goal.** Run 4 shows that the 10× result was context-specific and that using the actual Q4 weight representation creates substantially more quality loss in the recurrent model than the FP32 quality tests captured.

The central research direction remains viable, but the next technical bottleneck is now clear: **depth-shared weight compression and practical-context KV efficiency**, not container overhead.
