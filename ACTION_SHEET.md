# LARC Action Sheet

Canonical technical record for **LARC — Local Adaptive Representation & Compute**. Update this file after every substantive project run. Raw benchmark outputs live under `benchmarks/`; this sheet records what each experiment actually establishes and what it does not.

## Project objective

Build a local-AI model storage/execution standard that can reduce both model payload and **peak resident inference memory** by roughly **10–30× versus Q4-class GGUF deployment**, at the same context length, while preserving useful model capability.

A result does not count merely because a `.larc` file is small. Track separately:

- complete file / weight payload,
- unique resident weights,
- KV cache,
- scratch/workspace,
- complete resident inference tensors,
- quality delta,
- speed/latency,
- baseline and context,
- evidence class.

Evidence classes:

- **L0** — container/codec integrity.
- **L1** — operator/kernel benchmark.
- **L2** — trained conformance language model.
- **L2C** — conventional independent-layer model trained first, then converted to LARC and recovered after conversion.
- **L3** — independently hosted pretrained LLM conversion.
- **L4** — measured target-hardware RAM/VRAM + throughput.

---

# Run 1 — 2026-08-08

## Objective

Determine whether a new representation can mechanically enter the requested 10–30× storage range at all.

## Work completed

Implemented:

- initial `.larc` container,
- row-Q4/Q8 factors,
- activation-subspace projection bundles,
- HRVQ64 progressive vector coding,
- synthetic operator benchmarks,
- initial specification and prior-art review.

### HRVQ64

Nominal rates were 0.1875 / 0.3125 / 0.4375 bpw for 1/2/3 stages, mechanically corresponding to ~24× / 14.4× / 10.3× versus a 4.5-bpw payload. Quality was unacceptable as a whole-weight representation; three-stage synthetic output NMSE remained roughly 0.49–0.67.

**Decision:** HRVQ may be useful for low-importance residual pages, not as the base model representation.

Artifact: `benchmarks/benchmark_hrvq_run1.json`.

### Activation-subspace projection bundles

Five 384×384 operators sharing an activation domain:

| representation | storage reduction vs row-Q4 | held-out output NMSE at 98% core activation energy |
|---|---:|---:|
| rank-10 Q4 factors | **23.10×** | 0.0260 |
| rank-10 Q8 factors | **13.47×** | 0.0200 |
| rank-19 Q4 factors | **13.47×** | 0.0280 |

Artifact: `benchmarks/benchmark_projection_run1.json`.

## Run 1 conclusion

The requested byte regime is structurally reachable under favorable activation structure, but extreme weight-only coding is too destructive. The project direction changed to **shared structure + activation-aware factors + runtime-native residuals**. No real-LLM or VRAM claim was made.

---

# Run 2 — 2026-08-08

## Objective

Attack the original goals that Run 1 did not satisfy:

1. direct compressed execution,
2. weight residency,
3. KV residency,
4. complete inference memory,
5. model quality,
6. random-access / bounded-memory format semantics,
7. conversion of a model not originally trained with LARC sharing,
8. external pretrained-model validation harness.

## 1. Native packed-Q4 CPU execution — PASS (L1)

Implemented:

- `runtime/larc_q4.h`
- `runtime/larc_q4.cpp`
- `larc/q4_runtime.py`
- `tests/native_q4_smoke.cpp`
- `tests/native_q4_bench.cpp`

The kernel computes projected `A(Bx)` directly from packed INT4 nibbles. It never reconstructs `W = AB` and needs only rank-sized temporary storage.

Correctness smoke:

- maximum absolute difference versus separately dequantized projected reference: **~2.29e-5**,
- rank-19 scratch: **76 bytes**.

Transformer-like 1536×576 / rank-32 microbenchmark:

- direct row-Q4 payload: 448,512 B,
- LARC factors: 40,064 B,
- resident reduction: **11.19×**,
- direct row-Q4 reference GEMV: ~438.9 µs,
- LARC projected GEMV: ~33.1 µs,
- speedup in this scalar CPU microbenchmark: **13.27×**,
- scratch: 128 B.

Artifact: `benchmarks/run2_native_q4_kernel.json`.

**Established:** compressed-domain CPU execution is real and can reduce both bytes moved and compute for sufficiently low rank.

## 2. GPU packed execution — SOURCE COMPLETE, L4 OPEN

Implemented `runtime/triton_q4.py`. The Triton kernel extracts nibbles inside the kernel and supports projected `A(Bx)` with rank-sized device scratch.

Current environment has no usable CUDA/Triton hardware. Therefore GPU source exists, but GPU correctness, speed, and VRAM are **not measured**.

## 3. Recursive/shared logical graph — PASS (L2 representation)

LARC v0.2 permits multiple logical depths to reference one physical Transformer bundle, with room for recursion/depth-specific adapters.

Exact conformance test using a trained 16-logical-depth recurrent LM:

- shared validation NLL: 0.6147951,
- explicitly duplicated 16-copy validation NLL: 0.6147951,
- max logit difference: **0.0**,
- FP32 unique-weight reduction: **13.552×**,
- Q4-style file reduction: **13.404×**.

Artifact: `benchmarks/run2_recurrent_conformance.json`.

This validates graph aliasing. It is not broad-intelligence evidence.

## 4. Latent 2-bit KV cache

Implemented `larc/latent_kv.py`.

Historical K/V are projected into learned latent bases; the cache stores low-bit latent coefficients rather than full historical K/V vectors.

### Initial row/row latent Q2

Controlled rank-16 / head-dim-64 attention test:

- attention-output NMSE: **0.00280**,
- modeled SmolLM2-shaped KV reduction at 2K: **15.52×**,
- at 8K: **15.88×**.

Artifact: `benchmarks/run2_latent_kv_synthetic.json`.

### KIVI-oriented latent Q2

Updated keys to per-latent-channel grouping and values to per-token quantization. Rank 16:

- attention-output NMSE: **0.00831**,
- modeled SmolLM2-shaped KV reduction at 2K: **18.96×**,
- at 8K: **19.50×**.

Artifact: `benchmarks/run2_kivi_latent_kv_synthetic.json`.

The orientation is informed by KIVI prior art; combining that orientation with a learned latent subspace is a LARC research hypothesis. Synthetic quality is not a real-LLM claim.

## 5. LARC-native end-to-end memory + quality conformance — PASS (L2)

Implemented `tools/recurrent_kv_endtoend.py`.

Configuration:

- autoregressive character LM,
- hidden 128,
- 4 heads,
- 16 logical depths,
- context 64,
- one physical shared block,
- latent KV rank 12,
- actual allocated packed uint8 Q2 cache tensors + FP16 metadata.

Baseline token-by-token inference matched full causal inference to **1.335e-5 max-logit error**.

Quality:

- baseline NLL: 2.01662,
- packed latent-Q2 NLL: 2.24911,
- NLL increase: **11.53%**,
- screening gate: ≤15% — **PASS**.

Memory:

| pool | baseline | LARC |
|---|---:|---:|
| Q4-style weights | 1,129,482 B | 77,322 B |
| KV | 524,288 B | 57,344 B + 1,728 B basis |
| scratch | 7,680 B | 7,680 B |
| **total** | **1,661,450 B** | **144,074 B** |

- weight reduction: **14.61×**,
- complete inference-tensor reduction: **11.53×**,
- ≥10× gate: **PASS**.

Artifact: `benchmarks/run2_recurrent_kv_endtoend.json`.

This is the first experiment where memory and quality gates passed together in an executable trained LM.

## 6. Post-training conversion from conventional independent layers — PASS (L2C)

This is the strongest Run 2 result.

Implemented `tools/posttrain_recursive_conversion.py`.

Procedure:

1. Train a conventional autoregressive teacher with **16 independent physical Transformer blocks**.
2. Freeze/evaluate it as the baseline.
3. Build a one-physical-block recursive student only **after** teacher pretraining.
4. Initialize the shared block from the arithmetic mean of teacher depth states.
5. Perform short post-conversion recovery uptraining.
6. Fit shared latent K/V bases.
7. Run incremental inference with an actually packed latent-Q2 cache.
8. Compare against the original conventional teacher at the same context.

Important observation: naïve sharing was not free. Before recovery, student NLL was **55.08** versus teacher **1.8324**. Short recovery was essential.

Final result:

- teacher NLL: **1.83244**,
- converted shared student before packed KV: **1.91852**,
- converted LARC + packed latent-Q2: **2.08580**,
- final NLL increase vs independently trained teacher: **13.83%**,
- ≤15% gate: **PASS**.

Memory at context 64:

- conventional teacher Q4-style weight payload: **1,129,482 B**,
- converted LARC Q4-style weight payload: **77,322 B**,
- weight reduction: **14.61×**,
- conventional FP16 KV: **524,288 B**,
- LARC packed latent Q2 KV: **65,536 B** + **2,304 B** shared basis,
- bounded scratch: **8,704 B**,
- conventional total: **1,662,474 B**,
- LARC total: **153,866 B**,
- complete reduction: **10.80×**,
- ≥10× gate: **PASS**.

Baseline incremental implementation matched full causal inference to **1.43e-5 max-logit error**.

Artifact: `benchmarks/run2_posttrain_conversion.json`.

### Interpretation

The 10× result no longer depends on training a model with shared weights from inception. A conventionally parameterized model can be trained first, collapsed to a recursive/shared LARC representation, recovered with additional training, and remain inside the current quality screening threshold in this controlled LM setting.

This is still a small controlled corpus/model and is **not equivalent to converting an independently hosted 135M+/1B+ general LLM**.

## 7. SmolLM2-shaped complete memory accounting — STRUCTURAL PASS, NOT L3/L4

For a 105 MB Q4_K_M weight baseline and FP16 KV, rank-16 KIVI-oriented latent KV gives the following exact accounting for designed LARC structures:

| profile | context | LARC weight payload | weight reduction | KV reduction | modeled total reduction |
|---|---:|---:|---:|---:|---:|
| 10x | 2K | 8.06 MB | 13.02× | 18.96× | **14.00×** |
| 10x | 8K | 8.06 MB | 13.02× | 19.50× | **16.26×** |
| 15x | 2K | 5.31 MB | 19.78× | 18.96× | **18.73×** |
| 20x | 2K | 3.43 MB | 30.60× | 18.96× | **24.34×** |
| 30x | 2K | 2.43 MB | 43.23× | 18.96× | **28.98×** |

Artifact: `benchmarks/run2_kivi_memory_plan_rank16.json`.

This proves the proposed data structures fit the requested memory envelope. It does **not** prove SmolLM2 quality or measured VRAM.

## 8. LARC v0.2 paged file format — PASS (L0)

Implemented `larc/paged_container.py` and `tests/test_paged_container.py`.

Implemented semantics:

- fixed 64-byte header,
- fixed 64-byte page records,
- default 4 KiB alignment,
- numeric research codec IDs,
- CRC32 per page,
- dependency groups,
- `REQUIRED`, `SHARED`, `REFINEMENT`, `STREAMABLE`, `KV_BASIS` flags,
- mmap random-access page views,
- unique resident-payload accounting.

Local round-trip/alignment/checksum test passed.

`docs/SPEC.md` is now LARC v0.2.

## 9. External pretrained-model validation — HARNESS READY, L3 OPEN

Implemented `tools/real_model_benchmark.py` for `HuggingFaceTB/SmolLM2-135M` with 10× / 15× / 20× / 30× profiles.

The harness measures baseline NLL/perplexity, captures calibration activations, replaces compatible operators, factors the tied vocabulary matrix, executes via low-bit runtime modules, and reports output quality and encoded bytes.

### Current infrastructure boundary

- Local execution environment cannot retrieve Hugging Face/Xet model payloads.
- GitHub Actions jobs failed **before any workflow step was allocated**, even after replacing all reusable actions with pure `run:` steps.
- A smaller external TinyStories-260K checkpoint is visible and its model card reports validation loss 1.2968, but its 1.06 MB checkpoint is also Xet-backed; direct binary retrieval is blocked here.

No external checkpoint conversion actually ran. L3 is therefore **unpassed**, not experimentally failed.

The GitHub workflow is retained as **manual-only** so it does not create a permanently failing PR check.

## 10. Original-goal status after Run 2

| requirement | best current evidence | status |
|---|---|---|
| 10–30× smaller representation | post-training conversion: **14.61×** Q4-style weights | **PASS at L2C; L3 open** |
| 10–30× less resident weight memory | native operator: **11.19×**; converted model: **14.61×** | **PASS at L1/L2C** |
| 10–30× less total inference memory | converted model: **10.80×**; native model: **11.53×** | **PASS at L2/L2C; GPU L4 open** |
| reasonable/comparable model quality | converted model final NLL **+13.83%** vs independent teacher | **PASS screening gate at L2C; broad intelligence open** |
| compressed-domain CPU kernel | measured packed-Q4 C++ | **PASS** |
| compressed-domain GPU kernel | Triton packed-Q4 source | **hardware validation open** |
| random-access new file/runtime standard | v0.2 paged mmap format | **PASS research implementation** |
| arbitrary independent pretrained LLM conversion | harness exists; weight retrieval blocked | **OPEN** |
| measured 10×+ real GPU VRAM reduction | no available GPU | **OPEN** |

## 11. Design direction after Run 2

### Promoted

1. **Recursive/shared physical bundles.** Modern recursive-Transformer work provides a credible path for large reductions without requiring every original layer to survive independently.
2. **Short recovery/uptraining after conversion.** The controlled conversion showed naïve layer averaging was unusable; recovery transformed NLL 55.08 into 1.92 before KV compression.
3. **Activation-aware projection factors.** Useful inside shared bundles and for matrices that cannot be tied directly.
4. **Latent low-bit KV.** Mandatory once weight memory falls by an order of magnitude.
5. **Direct packed kernels.** Dense reconstruction is non-conforming for the target execution profile.
6. **Paged memory budgets.** File size and resident device memory must remain separate first-class metrics.

### Deprioritized

- whole-model sub-0.5-bpw raw-weight vector coding as the main representation,
- decompress-to-dense execution,
- claims based only on synthetic compression ratio.

## 12. Next controlling milestones

### P0 — L3 independently hosted pretrained model

- obtain an accessible checkpoint,
- compare against named Q4_K_M baseline at identical context,
- convert tensor/shard-wise,
- test shared-block grouping + depth adapters + projection residuals,
- run perplexity and external task/generation evaluation,
- sweep 10× / 15× / 20× / 30× and reject profiles crossing the quality threshold.

### P0 — real latent-KV validation

- capture real per-layer/head K/V spectra,
- fit latent bases on calibration data,
- evaluate long-context perplexity/generation,
- implement direct packed-Q2 latent-attention kernels so historical K/V are never reconstructed.

### P0 — L4 hardware

- CPU: measure peak RSS and tokens/s against llama.cpp Q4,
- NVIDIA: measure peak allocated/reserved VRAM and throughput using Triton/CUDA,
- Apple Silicon: add Metal packed-Q4/latent-KV kernels and measure unified-memory peak.

### P1 — conversion quality

Use relaxed-recursive/cross-layer-sharing methods rather than raw averaging for serious pretrained models:

`full pretrained stack → shared block groups → depth-wise adapters → calibration/uptraining/distillation → low-bit factors/residuals`.

## Run 2 key artifacts

- `docs/SPEC.md`
- `larc/paged_container.py`
- `larc/q4_runtime.py`
- `larc/latent_kv.py`
- `runtime/larc_q4.h`
- `runtime/larc_q4.cpp`
- `runtime/triton_q4.py`
- `tools/recurrent_kv_endtoend.py`
- `tools/posttrain_recursive_conversion.py`
- `tools/real_model_benchmark.py`
- `tools/memory_plan.py`
- `benchmarks/run2_posttrain_conversion.json`
- `benchmarks/run2_recurrent_kv_endtoend.json`
- `benchmarks/run2_native_q4_kernel.json`
- `benchmarks/run2_kivi_latent_kv_synthetic.json`
- `benchmarks/run2_kivi_memory_plan_rank16.json`

## Current conclusion

LARC has now demonstrated the complete requested **10× class memory reduction + bounded quality loss** in two executable controlled language-model settings, including one where the baseline model was conventionally pretrained with independent layers and compressed only afterward. The CPU packed-domain execution path and paged format are implemented.

The project is **not yet entitled to claim that arbitrary real-world GGUF models use 10–30× less GPU VRAM**. The remaining proof is L3/L4: an independently hosted general pretrained checkpoint and measured target-hardware GPU/Metal memory. Those are the controlling next milestones.
