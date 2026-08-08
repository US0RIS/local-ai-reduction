# local-ai-reduction / LARC

**LARC (Local Adaptive Representation & Compute)** is an experimental local-AI storage/execution standard aimed at reducing complete inference-memory cost, not merely model-file bytes.

## Target

Research target: **10–30× lower peak resident inference memory than a named Q4-class baseline at the same context length while retaining useful capability.**

## Current audited status — Run 4

The complete target is **not currently passed**. A second external audit found two important limitations in the prior controlled result:

1. the latent-value basis also requires a pseudoinverse / inverse-Gram correction after Q4 quantization;
2. the old quality path used FP32 weights while memory accounting charged Q4 weights.

Run 4 implements the value correction, exact basis-scale accounting, a full context sweep, and a Q4-weight quality control.

### Context dependence

For the controlled 16-depth recurrent/post-training geometry, with row/row latent-Q2 KV, Q4 bases, both FP16 inverse-Gram matrices, and Q4-style weight accounting:

| context | modeled total reduction |
|---:|---:|
| 64 | **10.52×** |
| 128 | **9.78×** |
| 512 | **8.65×** |
| 2K | **8.18×** |
| 8K | **8.05×** |

These are **modeled inference-tensor bytes, not measured RAM/VRAM**. The old 10.66× headline was therefore context-specific.

### Q4 weight quality

A canonical Run-4 reconstruction of the documented controlled protocol, evaluated over the same seed-999 100,032-character stream, found:

| path | FP32 NLL | dequantized-Q4 NLL | Q4 damage |
|---|---:|---:|---:|
| 16 independent teacher blocks | 1.81268 | 2.04418 | +0.23149 nats/char |
| one recurrent converted block | 2.01538 | 2.45371 | +0.43832 nats/char |

The reused block therefore suffers about **+0.20683 nats/char extra Q4 degradation** beyond the teacher's own Q4 loss. Correlated depth-wise quantization error is now a primary bottleneck.

### Equal-compute control

On the same 100,032-character evaluation stream, the reconstructed finite-step control still favors conversion/recovery:

- converted/recovered student: **2.01538 NLL**;
- recurrent model trained from scratch for the same 320 optimizer-step budget: **2.92223 NLL**.

This is an early-training/optimization result, not a convergence result.

### Reproducibility correction

The archived Run-3 100k artifact has no committed canonical generator. A Run-4 reconstruction with the documented seeds/protocol does not reproduce its exact NLL values, so Run-3's exact headline is now historical rather than promoted evidence.

Run 4 adds:

- `benchmarks/INDEX.json` provenance registry;
- `tools/check_benchmark_provenance.py`;
- a PR provenance workflow;
- committed generators for current Run-4 artifacts.

### Native factor fidelity

A redesigned low-noise rank-32 benchmark isolates factor quantization much better:

- **12.062×** resident factor reduction;
- theoretical rank-32 source floor: 0.00230 NMSE;
- measured projected-Q4 NMSE vs exact FP32: **0.03330**.

The projection architecture is mechanically effective, but current Q4 factor fidelity is not yet strong enough.

## SmolLM2 structural accounting only

After adding value inverse-Gram and basis-scale bytes, rank-16 KIVI-style latent KV is modeled at:

- **18.245×** smaller than FP16 KV at 2K;
- **19.309×** at 8K.

The nominal 10x weight profile is modeled at **13.87× total at 2K** and **16.17× at 8K**. No SmolLM2 quality result exists, so these are arithmetic feasibility results only.

## Still open

- fix recurrent/shared weight quantization, likely with QAT/recovery, depth adapters, residual rescue, or higher precision for sensitive shared operators;
- reduce KV metadata so practical-context total memory can exceed 10×;
- converged multi-seed equal-compute study;
- regenerate/promote all benchmark evidence only from committed generators;
- real activation spectra on an accessible pretrained Transformer;
- integrated packed runtime with measured RSS;
- **L3** external pretrained 135M+ model conversion;
- **L4** CUDA/Metal measured memory and throughput;
- competitive iso-byte comparison against GGUF IQ/K quants, AQLM/QuIP#-class methods, and smaller dense models.

## Repository map

- `ACTION_SHEET.md` — canonical current technical status.
- `benchmarks/RUN4_STATUS.json` — machine-readable Run-4 gate status.
- `benchmarks/INDEX.json` — artifact provenance registry.
- `docs/RUN4_AUDIT_PLAN.md` — second-audit closure plan.
- `larc/latent_kv.py` — corrected key/value basis-metric logic.
- `larc/q4_runtime.py` — canonical row-Q4 reference.
- `runtime/larc_q4.{h,cpp}` — native packed-Q4 CPU primitive.
- `tests/native_q4_fidelity.cpp` — low-noise projected-operator fidelity test.
- `tools/run4_control_reproduction.py` — committed control/Q4-quality generator.
- `tools/run4_context_sweep.py` — deterministic context-memory generator.

## Claim boundary

LARC currently demonstrates useful **mechanisms**—paged structural storage, physical parameter aliasing, direct packed execution, latent KV, and activation-subspace factors—but it does **not** yet demonstrate 10–30× less measured RAM/VRAM for a real pretrained local LLM with comparable quality.
