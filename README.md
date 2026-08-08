# local-ai-reduction / LARC

**LARC (Local Adaptive Representation & Compute)** is an experimental local-AI model representation/runtime focused on reducing complete inference-memory cost rather than download size alone.

## Target

Research target: **10–30× lower peak resident inference memory than a named Q4-class baseline at the same context length while retaining useful capability.** This is not yet proven on an external pretrained LLM or measured GPU/Metal VRAM.

## Audited Run 4 status

Run 4 makes the controlled quality and memory representations consistent: both teacher and converted model execute canonical Q4-dequantized weights, latent K/V bases are actually Q4, both K and V carry FP16 inverse-Gram corrections, compression calibration is disjoint from final evaluation, and historical artifacts are explicitly marked current/superseded/revoked.

Controlled synthetic character-LM result at **context 64**, evaluated on 100,032 characters:

| metric | result |
|---|---:|
| independent Q4 teacher NLL | **1.88548** |
| Q4-recovered shared model NLL | **1.94078** |
| shared + latent Q2 + E4M3 metadata NLL | **1.97525** |
| total quality delta | **+0.08977 nats/char** |
| perplexity ratio | **1.09392×** |

Training/checkpoint-selection/basis-calibration/final-evaluation streams use separate deterministic seeds (3/444/555/333).

The Q2 cache uses 2-bit latent coefficients plus one E4M3-FN min and scale byte per vector. At rank 16/head-dim 32 its raw K+V payload is **10.667× smaller than FP16 K+V**.

## Direct packed attention

`runtime/larc_q2_attention.cpp` consumes historical Q2 K/V directly from packed storage with Q4 bases and both inverse-Gram metrics. It does not materialize FP32 `T×rank` latent history.

At T=2048/rank16/head-dim32, the native correctness test reports:

- max abs error vs separately decoded reference: **2.5e-9**,
- packed cache per head: **24,576 B**,
- direct scratch per head: **8,448 B**,
- FP32 decoded K+V latent history: **262,144 B**.

Combining the controlled codec/model result with this separately validated packed-execution scratch contract gives **modeled** total tensor ratios of roughly:

- context 64: **12.04×**,
- context 2K: **10.60×**,
- context 8K: **10.50×**.

Only context 64 has quality validation. These are **not measured process RSS/VRAM**.

## Important negative result

Without Q4-aware recovery, recursively reusing one Q4-quantized block was highly destructive: compressed perplexity was ~**2.19×** the Q4 teacher. Run 4 therefore adds projected-Q4 recovery, reprojecting weights to the exact storage grid after every optimizer step. Extra recovery compute is part of the method and must be disclosed.

The previous Run-3 L2C headline is revoked because its quality path used FP32 weights while its memory path charged Q4 weights. The previous equal-compute control is also revoked as convergence evidence because the chosen continuation schedule degraded the independent teacher.

## SmolLM2 status

A SmolLM2-135M structural planner exists and correctly uses its GQA geometry. After current basis/metric accounting, rank-16 KIVI-shaped KV is modeled at ~18.25× smaller than FP16 KV at 2K and ~19.31× at 8K; the nominal `10x` profile models ~13.87× / 16.17× total at those contexts.

**No SmolLM2 quality result exists yet.** External checkpoint retrieval remains blocked in the available execution environment.

## Evidence and reproduction

- `ACTION_SHEET.md` — canonical technical record and open gates.
- `docs/SPEC.md` — v0.3-candidate semantics.
- `docs/RUN4_REPRO.md` — heavy controlled benchmark protocol.
- `benchmarks/ARTIFACT_MANIFEST.json` — current/historical/superseded/revoked artifact authority.
- `tools/run4_l2c_repro.py` — checkpointed heavy reproducer.
- `tools/check_quick_benchmark_artifacts.py` — deterministic quick artifact verifier.
- `runtime/larc_q4.{h,cpp}` — packed Q4 CPU primitives.
- `runtime/larc_q2_attention.{h,cpp}` — packed latent-Q2 attention primitive.
- `runtime/triton_q4.py` — CUDA/Triton reference source; hardware validation open.

Quick checks:

```bash
python -m pip install -e . pytest torch
pytest -q
python tools/check_quick_benchmark_artifacts.py

g++ -O3 -std=c++17 runtime/larc_q4.cpp tests/native_q4_fidelity.cpp -o /tmp/q4-fidelity && /tmp/q4-fidelity
g++ -O3 -std=c++17 runtime/larc_q4.cpp runtime/larc_q2_attention.cpp tests/native_q2_attention.cpp -o /tmp/q2-attn && /tmp/q2-attn
```

## Still open

- L3 independent pretrained 135M+ conversion with standard perplexity/tasks/generation.
- L4 measured CPU RSS / CUDA / Metal memory and optimized throughput.
- Real activation spectra and vocabulary-factorization validation.
- Long-context quality, not just long-context byte accounting.
- Stable converged equal-compute comparisons across multiple seeds.
- Competitive iso-byte baselines versus GGUF IQ/K-quants, AQLM/QuIP# where runnable, and smaller dense models.
- Real-model 20–30× quality retention.
