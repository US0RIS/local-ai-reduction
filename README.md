# local-ai-reduction / LARC

**LARC (Local Adaptive Representation & Compute)** is an experimental local-AI model representation/runtime aimed at reducing complete inference-memory cost, not merely file bytes.

## Target

Research target: **10–30× lower peak resident inference memory than a named competitive Q4-class baseline at the same context length while retaining useful capability.**

## Current audited status — Run 5

The strongest evidence is still controlled. LARC has **not** demonstrated 10–30× lower measured RAM/VRAM for a real pretrained GGUF model.

### Preferred controlled candidate

A 16-independent-block character LM is trained first, then converted to one shared physical block using:

- 80-step **teacher-layer function prefit** across all 16 teacher layer roles;
- 200-step **hard-projected group-64 Q4 QAT** recovery;
- signed `[-8,7]` Q4 weights with one FP16 scale per <=64 contiguous weights;
- rank-16 latent Q2 K/V;
- E4M3-FN min/scale metadata per latent vector;
- deterministic Q4 K/V bases;
- FP16 ridge-stabilized inverse-Gram correction for both K and V.

Five training seeds (`3,7,11,19,23`) are evaluated on the same independently generated **100,032-character** stream at context 64; basis calibration uses a disjoint seed-555 stream.

Against the project's **simple canonical row-Q4 teacher** reference:

- mean delta: **+0.00264 nats/char**;
- sample std: **0.15228**;
- mean perplexity ratio: **1.01208×**;
- PPL-ratio sample std: **0.15623**.

Against each FP32 teacher, mean perplexity ratio is **1.33287×**.

Artifact: `benchmarks/run5_e4m3_multiseed.json`.

This baseline is **not llama.cpp Q4_K_M**. The near-parity result is only versus the project's relatively weak row-Q4 reference.

### Modeled packed-memory contract

Using group-64 shared-weight bytes plus the direct-packed Q2/E4M3 attention scratch contract:

| context | modeled total reduction |
|---:|---:|
| 64 | **11.825×** |
| 256 | **11.123×** |
| 512 | **10.856×** |
| 1K | **10.682×** |
| 2K | **10.582×** |
| 4K | **10.527×** |
| 8K | **10.499×** |

Only context 64 has quality validation. These values are **modeled inference-tensor bytes, not measured process RSS or VRAM**.

Artifact: `benchmarks/run5_packed_context_sweep.json`.

### Native packed primitives

Two relevant CPU primitives are now separately validated:

**Packed Q2/E4M3 latent attention** (`runtime/larc_q2_attention.cpp`): at T=2048/rank16/head-dim32, max abs error versus a separately decoded reference is **2.50e-9**.

**Group-64 packed Q4 GEMV** (`runtime/larc_q4.cpp::q4_grouped_gemv`): on a 7×130 test matrix with a partial third group, max abs error is **3.34e-6**, and packed storage is exactly **497 B** as predicted.

Artifacts: `benchmarks/run4_native_q2_attention.json`, `benchmarks/run5_native_q4_group64.json`.

The primitives are **not yet integrated into one full-model native inference loop**. Run-5 quality uses mathematically equivalent reference/dequantized execution of the same representations.

### What the Run-5 audit changed

- The older `2.45371` Q4 result did not include latent-Q2 KV; complete-stack quality is now measured directly.
- A stochastic dither experiment did **not** support depth-correlated quantization error as the main failure mechanism.
- Shared-block rows have much larger absmax/RMS and >2× raw row-Q4 matrix NMSE than independent teacher blocks; finer scale grouping is better supported.
- Five training seeds replaced single-seed engineering conclusions.
- Function-space prefit + hard QAT substantially improved the 16→1 structural collapse.
- Grouped FP16 KV metadata was tested and remains an alternate path, but the existing E4M3 per-vector codec produced better five-seed quality and already has a native packed attention primitive.
- Context-dependent scratch and physical-vs-logical basis sharing are accounted separately.

Detailed audit response: `docs/RUN5_AUDIT_RESPONSE.md`.

## Still open — decisive gates

1. **Integrate the native primitives** into one full inference path and measure actual RSS/throughput.
2. **Real activation spectra** on an independently pretrained Transformer; this determines whether aggressive projection ranks transfer.
3. **Competitive baseline:** actual Q4_K_M/IQ or equivalent optimized runtime at matched context and quality.
4. **Long-context quality:** validate 256→8K, not only byte accounting.
5. **Convergence study:** tuned multi-seed teacher/shared/smaller-model learning curves.
6. **L3:** independent pretrained 135M+ conversion with standard perplexity/tasks/rare-token evaluation.
7. **L4:** measured CUDA/Metal/CPU peak memory, TTFT, and tokens/s.
8. **20–30×:** pursue after 10× passes the real-model/hardware gates.

## Repository map

- `ACTION_SHEET.md` — canonical current technical status.
- `benchmarks/RUN5_FINAL_STATUS.json` — machine-readable Run-5 status.
- `benchmarks/INDEX.json` — artifact provenance registry.
- `docs/RUN5_AUDIT_RESPONSE.md` — detailed Run-5 audit disposition.
- `tools/run5_e4m3_multiseed.py` — five-seed preferred-codec quality generator.
- `tools/run5_packed_context_sweep.py` — preferred packed byte model.
- `runtime/larc_q4.{h,cpp}` — row/group-Q4 native CPU primitives.
- `runtime/larc_q2_attention.{h,cpp}` — packed latent-Q2/E4M3 attention primitive.

## Claim boundary

> In a synthetic controlled L2C experiment, the preferred Run-5 representation models **11.825× lower inference-tensor residency at context 64 and 10.499× at 8K** than the project's internal row-Q4 reference. At context 64 across five training seeds, mean perplexity is **1.012×** that same reference. Separate native primitives validate the weight and KV arithmetic, but they are not yet integrated into one measured full-model runtime.

**Do not state that LARC has demonstrated 10–30× lower measured RAM/VRAM for real pretrained GGUF models.**
