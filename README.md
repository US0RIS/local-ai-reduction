# local-ai-reduction / LARC

**LARC (Local Adaptive Representation & Compute)** is an experimental local-AI model representation/runtime aimed at reducing complete inference-memory cost, not merely file bytes.

## Target

Research target: **10–30× lower peak resident inference memory than a named Q4-class baseline at the same context length while retaining useful capability.**

## Current audited status — Run 4

The strongest current evidence is still controlled, not a real pretrained-model or measured-VRAM result.

### Representation-consistent controlled quality

Synthetic character LM, **context 64**, 100,032 final-evaluation characters. Teacher and converted model both execute the canonical Q4 weight representation charged by memory accounting. Training, checkpoint selection, latent-basis calibration, and final evaluation use disjoint streams.

| path | NLL |
|---|---:|
| independent Q4 teacher | **1.88548** |
| Q4-recovered shared model | **1.94078** |
| shared + rank-16 latent Q2/E4M3 + Q4 K/V bases + both inverse-Gram metrics | **1.97525** |

Total degradation: **+0.08977 nats/char**, perplexity ×**1.09392**.

Artifact: `benchmarks/run4_fp8meta_l2c.json`; generator: `tools/run4_l2c_repro.py`.

### Direct packed latent-Q2 attention

`runtime/larc_q2_attention.cpp` consumes packed Q2 historical K/V, E4M3-FN min/scale metadata, Q4 bases, and both FP16 inverse-Gram metrics without constructing FP32 historical `T×rank` arrays.

At T=2048/rank16/head-dim32:

- max abs error vs separately decoded reference: **2.50e-9**;
- packed cache/head: **24,576 B**;
- direct scratch/head: **8,448 B**;
- FP32 decoded latent K+V history/head: **262,144 B**.

Artifact: `benchmarks/run4_native_q2_attention.json`.

### Context-indexed modeled tensor residency

Using the direct-packed scratch contract:

| context | modeled reduction |
|---:|---:|
| 64 | **12.04×** |
| 256 | **11.22×** |
| 512 | **10.91×** |
| 1K | **10.71×** |
| 2K | **10.60×** |
| 4K | **10.53×** |
| 8K | **10.50×** |

Only context 64 has quality validation. These numbers are **modeled inference-tensor bytes, not measured process RAM/VRAM**.

Artifact: `benchmarks/run4_packed_attention_context_sweep.json`; generator: `tools/run4_packed_context_sweep.py`.

## Important negative results

- Run-3's FP32-quality/Q4-memory headline is revoked.
- Naively applying Q4 to one physical block reused 16 times causes severe correlated error; projected-Q4 recovery is required.
- The old equal-compute control is not convergence evidence; stable tuned multi-seed convergence curves remain open.
- Historical benchmark artifacts that lack a canonical generator are not promoted evidence.

## SmolLM2

The structural planner correctly uses SmolLM2-135M GQA geometry. Existing structural arithmetic remains promising, but **no SmolLM2 quality benchmark has run** because checkpoint bytes remain inaccessible in the current execution environment.

## Evidence / reproduction

- `ACTION_SHEET.md` — canonical technical status.
- `benchmarks/INDEX.json` — artifact provenance registry.
- `benchmarks/RUN4_FINAL_STATUS.json` — current machine-readable claim boundary.
- `tools/check_benchmark_provenance.py` — provenance gate.
- `tools/run4_l2c_repro.py` — checkpointed controlled L2C reproducer.
- `tools/run4_packed_context_sweep.py` — packed runtime byte model.
- `runtime/larc_q4.{h,cpp}` — canonical packed-Q4 CPU primitives.
- `runtime/larc_q2_attention.{h,cpp}` — packed latent-Q2 attention primitive.

## Still open

- real activation spectra;
- long-context quality;
- converged multi-seed equal-compute controls;
- integrated full packed runtime with measured RSS;
- L3 independent pretrained model;
- L4 CUDA/Metal/CPU measured peak memory and optimized throughput;
- competitive iso-byte baselines;
- real-model 20–30× quality retention.

**Do not state that LARC has demonstrated 10–30× lower measured RAM/VRAM for real pretrained GGUF models.**
