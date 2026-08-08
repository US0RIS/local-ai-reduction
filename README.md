# local-ai-reduction / LARC

**LARC (Local Adaptive Representation & Compute)** is an experimental local-AI storage/execution standard aimed at reducing complete inference-memory cost, not merely model-file bytes.

## Target

Research target: **10–30× lower peak resident inference memory than a named competitive Q4-class baseline at the same context length while retaining useful capability.**

## Current audited status — Run 5

Run 5 responds to a third external technical audit and re-establishes a **controlled L2C >10× modeled tensor-memory result**, but only against the project's own simple row-Q4 baseline. It is **not** yet parity with llama.cpp Q4_K_M, an external pretrained model result, or measured RAM/VRAM.

### Current controlled candidate

A 16-independent-block character LM is trained first, then converted after training to one shared physical block using:

- 80-step teacher-layer function prefit across all 16 layer roles;
- 200-step hard-projected quantization-aware recovery;
- signed group-64 Q4 weights with FP16 sub-row scales;
- rank-16 latent-Q2 KV;
- one FP16 min/scale pair per 3-token K group and 3-token V group;
- Q4 K/V bases shared by the one physical recurrent block;
- FP16 ridge-stabilized inverse-Gram corrections for both K and V;
- explicitly charged FP16 incomplete-group tail;
- context-dependent reference workspace.

### Modeled memory

Baseline: **project row-Q4 teacher + FP16 KV + identical reference workspace**.

| context | modeled total reduction |
|---:|---:|
| 64 | **11.30×** |
| 128 | **11.19×** |
| 512 | **10.99×** |
| 2K | **10.89×** |
| 8K | **10.86×** |

These are structural tensor bytes, **not measured process RAM or VRAM**.

Artifact: `benchmarks/run5_memory_context.json`.

### Five-seed full-stack quality

Training seeds: `3, 7, 11, 19, 23`; same independently generated 100,032-character evaluation stream for every seed.

The LARC quality path executes the exact group-64-Q4 + grouped latent-Q2 representation whose bytes are charged.

Against the **same project row-Q4 teacher representation used for memory**:

- mean delta: **+0.03551 nats/char**;
- sample std: **0.16078 nats/char**;
- mean perplexity ratio: **1.04705×**;
- perplexity-ratio sample std: **0.17120**;
- range: **0.8969×–1.2363×**.

Against the FP32 teacher, the mean perplexity ratio is **1.37724×**.

Artifact: `benchmarks/run5_fullstack_multiseed.json`.

The distinction matters: the current controlled result is approximately at parity **with this project's primitive row-Q4 baseline on average**, not with optimized Q4_K_M and not with FP32.

### What the audit changed

- The Run-4 `2.45371` result did not include latent-Q2 KV; Run 5 now measures the complete stack directly.
- A stochastic dither diagnostic did **not** support depth-correlated quantization error as the main weight problem.
- Shared-block rows have much larger absmax/RMS and >2× raw row-Q4 weight NMSE than teacher blocks, motivating finer group-64 scale locality.
- Five seeds replaced single-seed conclusions.
- KV min/scale metadata is now grouped as a rate-distortion parameter; group 3 is the current controlled compromise.
- Scratch/workspace scales with context instead of being held constant.
- Both K and V basis corrections use the same ridge-stabilized inverse-Gram rule.
- A naive teacher-320 constant-LR run was not a convergence ceiling; a tuned convergence study remains open.
- The tiny controlled model's KV geometry does not upper-bound SmolLM2 because rank/head-dimension ratios differ materially.

Detailed response: `docs/RUN5_AUDIT_RESPONSE.md`.

## Still open — decisive gates

1. **Real activation spectra** on an independently pretrained Transformer. This decides whether aggressive projection ranks transfer beyond the synthetic task.
2. **Competitive baseline:** actual Q4_K_M/IQ or equivalent optimized deployment, not the project row-Q4 reference.
3. **Convergence study:** tuned multi-seed teacher/shared/smaller-model learning curves.
4. **Complete committed-generator replay** of the five-seed Run-5 artifact in an environment without the current execution ceiling.
5. **Integrated packed runtime + measured RSS.** Current memory remains structural accounting.
6. **L3:** external pretrained 135M+ conversion with standard perplexity/task/rare-token evaluation.
7. **L4:** CUDA/Metal measured VRAM, TTFT and tokens/s.
8. **20–30×** retained-quality regime after the 10× target passes L3/L4.

## Repository map

- `ACTION_SHEET.md` — canonical current technical status.
- `benchmarks/RUN5_FINAL_STATUS.json` — machine-readable Run-5 status.
- `benchmarks/INDEX.json` — artifact provenance registry.
- `docs/RUN5_AUDIT_RESPONSE.md` — full Run-5 audit disposition.
- `larc/grouped_kv.py` — grouped latent-Q2 storage/accounting primitives.
- `larc/latent_kv.py` — latent KV basis/metric logic.
- `runtime/larc_q4.{h,cpp}` — native packed-Q4 CPU primitive.
- `tools/run5_fullstack_protocol.py` — canonical five-seed protocol source.
- `tools/run5_fullstack_protocol_fp16tail.py` — exact FP16-tail semantics wrapper.
- `tools/run5_memory_sweep.py` — Run-5 context accounting.

## Claim boundary

> LARC has controlled five-seed L2C evidence for **10.86–11.30× lower modeled inference-tensor memory** than the project's simple row-Q4 baseline across context 64–8192, with mean perplexity ratio **1.047×** against that same baseline. Absolute mean perplexity remains **1.377×** the FP32 teacher. This is synthetic character-LM evidence, not Q4_K_M parity, not measured RAM/VRAM, and not evidence for arbitrary pretrained GGUF models.
