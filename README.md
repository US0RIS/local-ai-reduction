# local-ai-reduction / LARC

**LARC (Local Adaptive Representation & Compute)** is an experimental model storage + execution standard for reducing the complete memory cost of local language-model inference, not just the download size of weight files.

The design combines:

- shared / recursive physical Transformer bundles,
- activation-subspace projected operators,
- progressive residual pages,
- latent low-bit KV caches,
- packed-domain CPU/GPU kernel contracts,
- mmap-compatible random-access pages,
- explicit resident-memory budgeting.

## Target

Research target: **10–30× lower stored and peak resident inference memory than a Q4-class GGUF baseline**, at the same context length, while retaining useful model capability.

A `.larc` runtime must not need to reconstruct the complete dense model before inference.

## Current status — v0.2

### Passed in executable conformance tests

A trained 16-depth recurrent language-model test with an actually packed latent-Q2 KV cache achieved:

- **14.61×** lower Q4-style weight storage/residency,
- **11.53×** lower complete inference-tensor memory,
- **11.53%** NLL increase versus the uncompressed-cache baseline (inside the predefined ≤15% screening gate).

A stronger post-training test first trained a conventional Transformer with **16 independent physical blocks**, then converted it into one shared LARC block and ran short recovery uptraining. Final result:

- teacher NLL: **1.8324**,
- converted LARC + packed latent-Q2 NLL: **2.0858**,
- quality delta: **+13.83% NLL** — PASS,
- weight reduction: **14.61×**,
- total inference-tensor reduction: **10.80×** — PASS.

Artifact: `benchmarks/run2_posttrain_conversion.json`.

### Packed CPU execution

The native C++ kernel executes projected Q4 factors directly from packed nibbles, with rank-sized scratch and no dense reconstructed weight matrix.

1536×576 / rank-32 operator microbenchmark:

- **11.19×** less resident weight payload,
- **13.27×** faster than the reference direct row-Q4 scalar kernel in this CPU microbenchmark,
- 128-byte rank scratch.

Artifact: `benchmarks/run2_native_q4_kernel.json`.

### KV cache

LARC v0.2 includes latent 2-bit KV and a KIVI-oriented variant (key channels / value tokens). In a controlled attention simulation, rank-16 KIVI-style latent Q2 yielded ~0.83% attention-output NMSE while the modeled SmolLM2-shaped KV payload was **18.96× smaller at 2K context** and **19.50× smaller at 8K**.

These KV quality numbers are synthetic until repeated on an external pretrained model.

### SmolLM2-shaped memory accounting

For the 10× profile with rank-16 KIVI-oriented latent KV, complete modeled memory is **14.00× lower at 2K context** and **16.26× lower at 8K** than a 105 MB Q4_K_M weight baseline plus FP16 KV.

This is exact byte accounting for the designed structures, **not measured SmolLM2 VRAM and not SmolLM2 quality evidence**.

## What is not yet proven

Do **not** interpret the current results as “arbitrary GGUF models now use 10–30× less VRAM.” The remaining decisive gates are:

- convert an independently hosted pretrained 135M+ LLM and retain comparable broad quality at ≥10×,
- measure actual peak CUDA/Metal VRAM on target hardware,
- validate the Triton packed-Q4 path on a real GPU,
- run external perplexity/task benchmarks rather than only controlled conformance corpora.

The repository contains a SmolLM2-135M benchmark harness, but this execution environment cannot retrieve the external Xet model payload and GitHub-hosted runner jobs failed before allocating any workflow steps.

## Repository map

- [`ACTION_SHEET.md`](ACTION_SHEET.md) — authoritative technical run log and goal-gate status.
- [`docs/SPEC.md`](docs/SPEC.md) — LARC v0.2 research specification.
- `larc/paged_container.py` — mmap-compatible v0.2 paged container.
- `larc/latent_kv.py` — latent Q2 KV codecs.
- `runtime/larc_q4.{h,cpp}` — native packed-Q4 CPU kernel.
- `runtime/triton_q4.py` — packed-Q4 Triton/CUDA reference kernel.
- `tools/posttrain_recursive_conversion.py` — conventional-teacher → LARC conversion test.
- `tools/real_model_benchmark.py` — external SmolLM2 validation harness.
- `benchmarks/` — raw machine-readable benchmark evidence.

## Reproducing the local conformance work

```bash
python -m pip install -e . pytest torch
pytest -q

# LARC-native end-to-end packed KV test
PYTHONPATH=. python tools/recurrent_kv_endtoend.py

# Pretrain conventional independent-layer teacher, convert, recover, and test
PYTHONPATH=. python tools/posttrain_recursive_conversion.py

# Synthetic projection / KV studies
PYTHONPATH=. python tools/benchmark_projection.py --n 384 --operators 5 --samples 768
PYTHONPATH=. python tools/benchmark_latent_kv.py

# Native C++ packed-Q4 checks
g++ -O3 -march=native -std=c++17 runtime/larc_q4.cpp tests/native_q4_smoke.cpp -o /tmp/larc-smoke
/tmp/larc-smoke

g++ -O3 -march=native -std=c++17 runtime/larc_q4.cpp tests/native_q4_bench.cpp -o /tmp/larc-bench
/tmp/larc-bench
```

## Why a new format rather than a smaller GGUF container?

The largest gains require changing the object being stored and executed. LARC can represent many logical layers with shared physical parameters, factor an operator into compressed stages, attach progressive corrections, keep historical attention state in a latent low-bit space, and explicitly page these components under a memory budget. A container restricted to independent fixed dense tensors cannot express all of those runtime semantics directly.
