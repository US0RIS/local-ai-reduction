# LARC Action Sheet

Canonical technical record for **LARC — Local Adaptive Representation & Compute**. Historical Run 1–3 details remain in `docs/RUN3_AUDIT_CORRECTIONS.md` and historical benchmark artifacts. Current artifact authority is `benchmarks/INDEX.json`; current machine-readable status is `benchmarks/RUN4_FINAL_STATUS.json`.

## Objective

Reduce local-LLM **peak resident inference memory by roughly 10–30× versus a named Q4-class baseline at the same context length**, while retaining useful capability. Every claim must name baseline, context, quality delta, byte pools, and evidence type.

Evidence levels: **L0** format, **L1** operator/runtime, **L2** controlled trained model, **L2C** controlled post-training conversion, **L3** independent pretrained LLM, **L4** measured target hardware.

---

# Run 4 — audited representation-consistent closure

## Audit corrections that survive

1. **Both latent K and V need inverse-Gram correction after Q4 basis quantization.** Current value reconstruction is `v_lat G_V^-1 B_V_hat`; scores use `G_K^-1`. Basis row scales and both FP16 metrics are charged.
2. **Canonical Q4_ROW** remains `q in [-8,7]`, low nibble first, FP16 row scale, `scale=max(max_positive/7, abs(min_negative)/8, eps)`.
3. **Quality and memory representations must match.** Run-3's FP32-weight quality / Q4-weight memory headline is revoked.
4. **Compression calibration must be disjoint from final evaluation.** Current streams are training seed 3, Q4 checkpoint-selection 444, latent-basis calibration 555, final evaluation 333.
5. **Context length is mandatory in every total-memory claim.**
6. **Artifact provenance is systemic.** `benchmarks/INDEX.json` plus `tools/check_benchmark_provenance.py` governs promoted evidence; new current artifacts name committed generators.

## Negative result: naive Q4 sharing fails

When the independent teacher and recursively shared model were both evaluated using the Q4 representation charged by the memory model, the unrecovered shared model degraded severely (approximately perplexity ×2.19 versus the Q4 teacher). This confirms that quantization error in one physical block reused across depth is highly correlated.

**Decision:** aggressive recursive sharing requires representation-aware recovery; FP32 recovery alone is insufficient.

## Projected-Q4 recovery

After structural conversion/recovery, matrices are projected to canonical Q4 and 1-D parameters to FP16. Optimization continues at low LR, and after every optimizer step parameters are immediately reprojected onto the exact storage grid. Checkpoint selection uses a disjoint stream.

Recovery provenance for the current controlled result:

- teacher pretraining: 120 steps,
- structural recovery: 200 steps,
- projected-Q4 recovery: up to 200 steps,
- selected projected-Q4 checkpoint: step 150.

Extra recovery compute is part of the method and must be disclosed.

## Current controlled quality result — L2C

Synthetic character LM, **context 64**, final evaluation **100,032 characters**:

| path | NLL |
|---|---:|
| independent canonical-Q4 teacher | **1.88548** |
| Q4-recovered shared model, normal KV | **1.94078** |
| shared + rank-16 latent Q2 + E4M3-FN metadata + Q4 bases + both metrics | **1.97525** |

Quality decomposition:

- structural/Q4-recovery penalty: **+0.05529 nats/char**,
- latent-KV penalty: **+0.03447 nats/char**,
- total: **+0.08977 nats/char**,
- perplexity ratio: **1.09392×**.

Artifact: `benchmarks/run4_fp8meta_l2c.json`. Generator: `tools/run4_l2c_repro.py`.

This is the strongest current **quality** result. It is a narrow synthetic controlled model, not broad-intelligence or external-pretrained evidence.

## E4M3-FN latent-Q2 metadata

Rank-16/head-dim-32 row-Q2 stores 4 coefficient bytes plus one E4M3-FN min and one scale byte per vector. K+V therefore cost **12 B/token** versus **128 B/token FP16**, a raw **10.6667× KV reduction** without lowering latent rank.

## Direct packed latent-Q2 attention — L1

Implemented `runtime/larc_q2_attention.{h,cpp}` and `q4_transposed_gemv`.

The CPU reference consumes packed Q2 K/V, E4M3 metadata, Q4 bases, and both inverse-Gram metrics without materializing FP32 historical `T×rank` K/V arrays.

At **T=2048, rank=16, head_dim=32**:

- max absolute error vs separately decoded reference: **2.50e-9**,
- packed cache/head: **24,576 B**,
- direct scratch/head: **8,448 B**,
- FP32 decoded latent K+V history/head: **262,144 B**.

Artifact: `benchmarks/run4_native_q2_attention.json`; generator: `tests/native_q2_attention.cpp`.

This is correctness/memory-contract evidence, not optimized throughput.

## Packed-runtime structural context sweep

Combining current Q4/shared weight bytes, E4M3 latent-Q2 cache bytes, and the direct-packed scratch contract gives:

| context | modeled total tensor reduction |
|---:|---:|
| 64 | **12.04×** |
| 256 | **11.22×** |
| 512 | **10.91×** |
| 1K | **10.71×** |
| 2K | **10.60×** |
| 4K | **10.53×** |
| 8K | **10.50×** |

Artifact: `benchmarks/run4_packed_attention_context_sweep.json`; generator: `tools/run4_packed_context_sweep.py`.

**Quality is validated only at context 64.** The 2K/8K rows are structural/runtime byte models, not long-context quality and not measured RSS/VRAM.

## Existing mainline Run-4 evidence preserved

Current `main` already contained:

- exact Q4 endpoint/scale semantics,
- both K/V inverse-Gram accounting,
- low-noise native factor-fidelity benchmark,
- context-sweep and SmolLM2 structural generators,
- `benchmarks/INDEX.json` provenance policy and CI gate,
- same-stream reconstruction diagnostics.

Those files remain in place; the promoted packed-runtime/L2C artifacts above are layered on top rather than replacing historical evidence.

## SmolLM2 status

The structural planner uses SmolLM2's GQA geometry (`kv_heads=3`, `head_dim=64`). Current structural arithmetic remains promising (>10× modeled total for aggressive profiles), but **no SmolLM2 quality benchmark has run**. External checkpoint payload access remains the L3 blocker in this environment.

---

# Current claim boundary

The strongest defensible statement is:

> In a controlled synthetic post-training language-model conversion at **context 64**, with teacher and LARC quality paths executing the same canonical Q4 weight representation and with disjoint training/selection/calibration/evaluation streams, LARC's rank-16 latent-Q2/E4M3 representation produces **+0.08977 nats/char** degradation (perplexity ×**1.09392**). Combining that L2C result with the separately L1-validated direct-packed attention scratch contract yields a **12.04× modeled same-context inference-tensor reduction**. The corresponding structural model remains ~10.60× at 2K and ~10.50× at 8K, but long-context quality is unvalidated.

Do **not** claim that LARC has demonstrated 10–30× lower measured RAM/VRAM for real pretrained GGUF models.

## Open hard gates

1. **Converged equal-compute, multi-seed control.** The earlier control is not convergence evidence; stable tuned schedules and teacher-at-budget ceiling are required.
2. **Real activation geometry.** Measure real Transformer activation spectra at candidate ranks.
3. **Long-context quality.** Validate the packed codec at 256→8K, not just byte accounting.
4. **Integrated packed full-model runtime.** Execute packed Q4 weights and packed latent KV together and measure actual RSS.
5. **L3 external pretrained model.** SmolLM2-135M or larger, standard perplexity/tasks/generation and rare-token checks.
6. **L4 hardware.** CUDA/Metal/CPU peak memory and optimized throughput against named baselines.
7. **Competitive iso-byte baselines.** GGUF IQ/K quants, AQLM/QuIP# where runnable, and smaller dense models.
8. **20–30× real-model quality.** Completely open.
