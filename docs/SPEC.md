# LARC v0.3-candidate — Local Adaptive Representation & Compute

**Status:** experimental research specification; v0.2 container framing retained  
**Extension:** `.larc`  
**Objective:** maximize retained model capability per **peak resident inference byte**, not merely file bytes.

LARC permits the logical Transformer graph to differ from a list of independently stored dense tensors. Logical operators can reference shared physical blocks, activation-subspace factors, progressive residuals, low-bit KV state, and random-access pages. A `DIRECT_PACKED` runtime executes stored representations without reconstructing the complete dense model.

## 1. Mandatory claim semantics

Every memory/compression claim MUST state:

1. exact baseline and codec,
2. **context length** and batch/generation mode,
3. complete serialized bytes,
4. unique resident weight bytes,
5. KV bytes,
6. scratch/workspace bytes,
7. modeled total inference-tensor bytes,
8. measured RSS/device peak when available,
9. quality delta in nats/token (or nats/character) plus perplexity ratio,
10. throughput/latency,
11. whether each number is measured, modeled, synthetic, or composed across evidence levels.

A GGUF Q4_K_M file size and LARC's internal `Q4_ROW` estimator are different baselines and MUST NOT be interchanged.

**Representation-consistency rule:** if a memory claim charges Q4 weights, the associated quality path MUST execute the same Q4-dequantized values or their direct-packed mathematical equivalent.

The research target remains 10–30× lower peak resident inference memory than a named Q4-class deployment at the same context while retaining useful capability.

## 2. Logical graph / physical bundles

The manifest separates `logical_nodes` from `physical_bundles`. Logical nodes may be `DENSE_REF`, `PROJECTION`, `RECURSIVE_REF`, `RELAXED_RECURSIVE_REF`, or `PROJECTION_RESIDUAL`.

A shared physical page is counted once. Exact aliasing is lossless only when the logical model genuinely reuses that parameter object. Collapsing independent pretrained layers into one shared bundle is lossy conversion and requires independent quality accounting and recovery provenance.

## 3. Canonical `Q4_ROW`

The v0.3-candidate storage contract is identical in Python/C++:

- integer range `q in [-8,7]`,
- nibble `code = q + 8`,
- low nibble first,
- odd-column padding code `8`,
- one IEEE-754 binary16 scale per output row,
- `scale = max(max(w,0)/7, max(-w,0)/8, epsilon)`,
- `q = clip(round(w/scale), -8, 7)`,
- `w_hat = (code - 8) * scale`.

This is the minimum positive scale that fits both signed extrema without deliberate clipping. Golden tests exercise both code 0 (`-8`) and code 15 (`+7`).

## 4. Projection bundles

For operators sharing an input domain, LARC stores `y_i ~= A_i(Bx)`.

The basis MUST be quantized before the final projected-operator fit when quality claims use quantized execution:

1. derive calibration basis `B`,
2. quantize/dequantize to `B_hat`,
3. form `Z = B_hat X`,
4. solve `min_A ||W X - A Z||_F^2` with documented regularization,
5. encode `A`.

A direct-packed runtime MUST NOT materialize `A B_hat`.

## 5. Progressive residuals / codebooks

Residual pages may contain sparse, low-rank, HRVQ64, or rotated low-bit corrections and SHOULD be ordered by validation gain per resident byte.

Any codebook bpw claim MUST include amortized codebook bytes unless the codebook is normatively shared and its sharing scope is stated. A 256×64 FP16 HRVQ codebook is 32,768 B/stage.

## 6. Latent KV

Historical K/V may be stored as rank-`r` latent coefficients:

`k_lat = B_k k`, `v_lat = B_v v`.

### 6.1 Q2 coefficients and metadata

`LATENT_Q2_ROW` stores asymmetric 2-bit coefficients per latent vector. Four coefficients share one byte.

Two metadata profiles are research-defined:

- **FP16 metadata:** one FP16 min + FP16 scale per vector (4 metadata bytes/vector),
- **E4M3-FN metadata:** one FP8 min + FP8 scale per vector (2 metadata bytes/vector).

Metadata format MUST be declared in the manifest. `LATENT_Q2_KIVI` may group key metadata over token groups while retaining per-token value metadata.

### 6.2 Quantized-basis pseudo-inverse metrics

A Q4-dequantized basis `B_hat` is not exactly row-orthonormal. **Both K and V paths require metric correction** unless an equivalent transform is folded into another stored operator.

Store, reference FP16:

`G_K^-1 = (B_K_hat B_K_hat^T + lambda I)^-1`

`G_V^-1 = (B_V_hat B_V_hat^T + lambda I)^-1`.

Score:

`score_t ~= q_lat^T G_K^-1 k_lat_t / sqrt(d_head)`.

Value reconstruction:

`v_out ~= (sum_t alpha_t v_lat_t) G_V^-1 B_V_hat`.

The latter is the row-vector form of the Moore-Penrose row-space reconstruction. Metric bytes and every Q4 basis row scale MUST be charged.

### 6.3 Calibration provenance

Current reference basis fitting uses deterministic uncentered eigendecomposition of `X^T X`, not randomized PCA. Calibration data MUST be disjoint from final evaluation data for promoted quality claims.

## 7. Direct packed execution

### 7.1 Q4 operators

`runtime/larc_q4.cpp` implements `q4_gemv`, `q4_transposed_gemv`, and projected `A(Bx)` without reconstructing complete dense weights.

### 7.2 Packed latent-Q2 attention

`runtime/larc_q2_attention.cpp` consumes:

- Q4 K/V bases,
- FP16 K/V inverse-Gram matrices,
- packed Q2 historical K/V,
- E4M3-FN min/scale metadata,

and computes one-head autoregressive attention without materializing historical latent K/V as FP32 `T×r` arrays.

Required reference scratch is `T + 4r` FP32 values for one head: attention scores plus rank-sized current/aggregate vectors. Scratch may be reused across heads.

`tests/native_q2_attention.cpp` compares direct packed execution with a separately decoded reference. The implementation is a correctness/memory primitive, **not an optimized throughput claim**.

CUDA/Triton Q4 source remains reference-only until hardware validation.

## 8. Paged file layout

The v0.2 framing remains: 64-byte little-endian header `<8sHHIQQQQQ8x`, 64-byte page records `<IHHQQQII24x`, default 4096-byte payload alignment, per-page CRC32, dependency groups, and `REQUIRED/SHARED/REFINEMENT/STREAMABLE/KV_BASIS` flags.

Header/page-table authentication and verify-on-open vs verify-on-touch policy remain open standards work.

## 9. Memory-budget execution

A runtime SHOULD pin required/shared pages, allocate the selected KV tier at the requested context, reserve bounded scratch, admit refinement pages by marginal validation gain/byte, prefetch streamable pages, evict refinements before core pages, and report unique resident pages/bytes.

A total-memory claim at one context MUST NOT be generalized to another context without a context sweep or measurement.

## 10. Conversion / evaluation provenance

Record source architecture, sharing map, initialization, recovery objective/data, optimizer schedule, steps, Q4-aware recovery if used, calibration data, checkpoint-selection data, and final evaluation data.

Training, checkpoint selection, compression calibration, and final evaluation SHOULD be disjoint for promoted held-out results.

## 11. Artifact provenance

`benchmarks/INDEX.json` is authoritative about promoted/historical evidence. Promoted artifacts MUST name a committed generator and pass `tools/check_benchmark_provenance.py`.

Current packed-runtime arithmetic is generated by `tools/run4_packed_context_sweep.py`; current controlled training is reproduced by `tools/run4_l2c_repro.py`.

A numerical artifact that cannot be reproduced from its declared generator MUST be revoked or demoted rather than silently retained as current evidence.

## 12. Validation levels

- **L0:** format/codec integrity and byte accounting.
- **L1:** operator/runtime correctness and explicit source/output error.
- **L2:** controlled trained conformance model.
- **L2C:** post-training conversion of an independently parameterized controlled model, with representation-consistent quality.
- **L3:** independently hosted pretrained LLM against a named deployment baseline and standard evaluation.
- **L4:** measured CPU/GPU/accelerator peak memory and throughput.

L0–L2C MUST NOT be presented as L3/L4.

## 13. Current audited boundary

At **context 64**, the current controlled L2C result is approximately:

- Q4 independent teacher NLL: **1.88548**,
- Q4-recovered shared model NLL: **1.94078**,
- shared + latent Q2/E4M3-FN metadata + both K/V metrics: **1.97525**,
- total delta: **+0.08977 nats/char**,
- perplexity ratio: **1.09392×**.

Combining this L2C representation with the separately L1-validated direct-packed attention scratch contract gives approximately **12.04× modeled same-context inference-tensor reduction** at context 64. The packed structural sweep is ~**10.60× at 2K** and ~**10.50× at 8K**, but quality is validated only at context 64.

These are controlled synthetic/modelled results. They are not measured whole-process RAM/VRAM, not external-pretrained-model evidence, and not proof of long-context quality. L3 and L4 remain open.
