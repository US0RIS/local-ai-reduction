# LARC v0.4-candidate — Local Adaptive Representation & Compute

**Status:** experimental research specification; v0.2 container framing retained  
**Extension:** `.larc`  
**Objective:** maximize retained model capability per **peak resident inference byte**, not merely file bytes.

LARC permits the logical Transformer graph to differ from a list of independently stored dense tensors. Logical operators can reference shared physical blocks, activation-subspace factors, progressive residuals, low-bit KV state, and random-access pages. A `DIRECT_PACKED` runtime executes stored representations without reconstructing the complete dense model.

## 1. Mandatory claim semantics

Every promoted memory/compression claim MUST state:

1. exact baseline/runtime/codec,
2. context length and batch/generation mode,
3. serialized bytes,
4. unique resident weight bytes,
5. KV bytes,
6. scratch/workspace bytes,
7. modeled total inference-tensor bytes,
8. measured RSS/device peak when available,
9. quality delta in nats/token or nats/character and perplexity ratio,
10. throughput/latency,
11. seed/evaluation coverage for learned-model claims,
12. whether each quantity is measured, modeled, synthetic, or composed across evidence levels.

A GGUF Q4_K_M deployment and LARC's internal `Q4_ROW` reference are different baselines and MUST NOT be interchanged.

**Representation-consistency rule:** quality MUST execute the representation whose bytes are charged, or a mathematically equivalent direct-packed implementation.

The research target remains 10–30× lower peak resident inference memory than a named competitive Q4-class deployment at the same context while retaining useful capability.

## 2. Logical graph / physical bundles

The manifest separates `logical_nodes` from `physical_bundles`. Logical nodes may be `DENSE_REF`, `PROJECTION`, `RECURSIVE_REF`, `RELAXED_RECURSIVE_REF`, or `PROJECTION_RESIDUAL`.

A shared physical page is counted once. Exact aliasing is lossless only when the logical model genuinely reuses that parameter object. Collapsing independently trained layers into one shared bundle is a lossy conversion and requires quality accounting and recovery provenance.

For latent KV, logical histories and physical basis objects are separate concepts. A recurrent model may have many logical K/V histories while sharing one physical K/V basis/metric set. Implementations and byte accounting MUST represent both counts explicitly.

## 3. Signed Q4 family

### 3.1 Canonical `Q4_ROW`

- integer range `q in [-8,7]`,
- nibble `code=q+8`,
- low nibble first,
- odd-column padding code `8`,
- IEEE-754 binary16 scale,
- `scale=max(max(w,0)/7, max(-w,0)/8, epsilon)`,
- `q=clip(round(w/scale),-8,7)`,
- `w_hat=(code-8)*scale`.

`Q4_ROW` stores one scale per output row.

### 3.2 `Q4_GROUP64`

Run-5 candidate weight codec. Each matrix row is partitioned into contiguous groups of at most 64 values; each group uses the same signed-Q4 rule independently and stores one FP16 scale.

Native reference type: `Q4GroupRows`.  
Native primitive: `q4_grouped_gemv` in `runtime/larc_q4.cpp`.

A conformance test MUST exercise a non-multiple-of-64 width so the partial final group is tested.

## 4. Projection bundles

For operators sharing an input domain, LARC may store `y_i ~= A_i(Bx)`.

The basis MUST be quantized before the final projected-operator fit when the stored basis is quantized:

1. derive calibration basis `B`,
2. encode/decode to actual `B_hat`,
3. form `Z=B_hat X`,
4. solve `min_A ||W X - A Z||_F^2` with documented regularization,
5. encode `A`.

A direct-packed runtime MUST NOT materialize `A B_hat`.

## 5. Progressive residuals / codebooks

Residual pages may contain sparse, low-rank, HRVQ64, or rotated low-bit corrections and SHOULD be ordered by marginal validation gain per resident byte.

Any codebook bpw claim MUST include amortized codebook bytes unless the codebook is normatively shared and its sharing scope is stated. A 256×64 FP16 HRVQ codebook is 32,768 B/stage.

## 6. Latent KV

Historical K/V may be projected into rank-`r` coordinates:

`k_lat=B_k k`, `v_lat=B_v v`.

### 6.1 Q2 coefficient storage

Four asymmetric 2-bit coefficients share one byte.

Research metadata profiles include:

- **FP16 per-vector:** FP16 min + FP16 scale,
- **E4M3-FN per-vector:** one FP8 min + one FP8 scale,
- **group-scalar:** one min/scale pair shared over an explicit token group and latent dimensions.

Metadata format, token-group size, and incomplete-group policy MUST be declared.

### 6.2 Basis pseudo-inverse metrics

A decoded low-bit basis `B_hat` is generally not exactly row-orthonormal. Unless an equivalent correction is folded elsewhere, both K and V use ridge-stabilized inverse Gram:

`G_K^-1=(B_K B_K^T + lambda I)^-1`

`G_V^-1=(B_V B_V^T + lambda I)^-1`.

Score:

`score_t ~= q_lat^T G_K^-1 k_lat_t / sqrt(d_head)`.

Value reconstruction:

`v_out ~= (sum_t alpha_t v_lat_t) G_V^-1 B_V`.

The current controlled ridge is `lambda = 1e-5 * mean(diag(B B^T))` per head. Metric and basis-scale bytes MUST be charged.

### 6.3 Calibration provenance

Promoted quality claims SHOULD use deterministic basis fitting and MUST keep compression calibration disjoint from final evaluation. The preferred packed path uses uncentered eigendecomposition of `X^T X`.

## 7. Direct packed execution

### 7.1 Q4 weights

`runtime/larc_q4.cpp` implements:

- `q4_gemv`,
- `q4_transposed_gemv`,
- projected `A(Bx)`,
- `q4_grouped_gemv` for group-scale Q4.

All consume packed nibbles directly.

### 7.2 Q2/E4M3 latent attention

`runtime/larc_q2_attention.cpp` consumes Q4 K/V bases, FP16 inverse-Gram matrices, packed historical Q2 K/V, and E4M3-FN min/scale metadata without materializing historical latent K/V as FP32 `T×r` arrays.

The one-head reference scratch contract is `T + 4r` FP32 values. Scratch may be reused across heads.

The native CPU primitive is correctness/memory-contract evidence, not optimized-throughput evidence.

### 7.3 Integrated runtime requirement

Separate native weight and attention primitives do **not** constitute an integrated full-model runtime. A full runtime gate requires the actual model to execute both primitive families in one inference process, followed by measured process/device memory and throughput.

## 8. Paged file layout

The v0.2 framing remains: 64-byte little-endian header `<8sHHIQQQQQ8x`, 64-byte page records `<IHHQQQII24x`, default 4096-byte payload alignment, per-page CRC32, dependency groups, and `REQUIRED/SHARED/REFINEMENT/STREAMABLE/KV_BASIS` flags.

Header/page-table authentication and verify-on-open vs verify-on-touch policy remain open standards work.

## 9. Memory-budget execution

A runtime SHOULD pin required/shared pages, allocate the selected KV tier at requested context, reserve context-dependent scratch, admit refinements by marginal validation gain/byte, prefetch streamable pages, evict refinements before core pages, and report unique resident pages/bytes.

A total-memory claim at one context MUST NOT be generalized to another context without a context sweep or measurement.

## 10. Conversion / evaluation provenance

Record source architecture, sharing map, initialization, function-prefit/distillation/recovery objective, recovery data, optimizer schedule, steps, quantization projection schedule, calibration data, checkpoint-selection data, seed list, and final evaluation data.

Promoted held-out results SHOULD keep training, selection, compression calibration, and final evaluation disjoint.

Run-5 controlled conversion uses teacher-layer function prefit followed by hard-projected `Q4_GROUP64` QAT recovery.

## 11. Artifact provenance

`benchmarks/INDEX.json` is authoritative about promoted/historical evidence. Promoted artifacts MUST name a committed generator when one exists. Multi-seed coverage and native-packed execution are separate evidence axes and MUST be reported separately.

A numerical artifact that cannot be reproduced from its declared generator MUST be revoked or demoted rather than silently retained as current evidence.

## 12. Validation levels

- **L0:** format/codec integrity and byte accounting.
- **L1:** operator/runtime correctness and explicit source/output error.
- **L2:** controlled trained conformance model.
- **L2C:** post-training conversion of an independently parameterized controlled model, with representation-consistent quality.
- **L3:** independently hosted pretrained LLM against a named competitive deployment baseline and standard evaluation.
- **L4:** measured CPU/GPU/accelerator peak memory and throughput.

L0–L2C MUST NOT be presented as L3/L4.

## 13. Current audited boundary — Run 5

Preferred controlled candidate:

- 16-independent-block teacher trained first;
- one recurrent physical block after conversion;
- 80-step teacher-layer function prefit;
- 200-step hard-projected `Q4_GROUP64` QAT recovery;
- rank16 latent Q2;
- E4M3-FN per-vector metadata;
- deterministic Q4 K/V bases;
- FP16 K/V inverse-Gram metrics.

Baseline is the project's **simple `Q4_ROW` teacher + FP16 KV**, not llama.cpp Q4_K_M.

At context 64 across five training seeds, 100,032 held-out characters/seed:

- mean delta vs internal row-Q4: **+0.00264 nats/char**,
- sample std: **0.15228**,
- mean perplexity ratio: **1.01208×**,
- sample std of ratio: **0.15623**,
- mean perplexity ratio vs FP32 teacher: **1.33287×**.

Combined modeled packed tensor reduction:

- context64: **11.825×**,
- context2K: **10.582×**,
- context8K: **10.499×**.

Quality is validated only at context64. Native L1 primitives separately validate `Q4_GROUP64` GEMV and Q2/E4M3 attention, but they are not yet integrated into one full-model native runtime.

These are controlled synthetic/modelled results. They are not Q4_K_M parity, not measured whole-process RAM/VRAM, and not external-pretrained-model evidence. L3 and L4 remain open.
