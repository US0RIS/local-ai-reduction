# LARC v0.3-candidate — Local Adaptive Representation & Compute

**Status:** experimental research specification; v0.2 container framing retained  
**Extension:** `.larc`  
**Primary objective:** maximize retained model capability per **peak resident inference byte**, not merely per file byte.

LARC is a runtime-oriented representation in which the logical Transformer graph can differ from a list of independently stored dense tensors. Logical operators may reference shared physical parameter bundles, activation-subspace factors, depth adapters, progressive residual pages, and compressed KV-cache bases. A conforming direct-packed runtime executes these representations without reconstructing the complete dense model.

## 1. Required reporting semantics

Every compression claim MUST identify its exact baseline and report separately:

1. complete serialized file bytes,
2. unique resident weight bytes,
3. KV-cache bytes at a stated context length,
4. scratch/workspace bytes,
5. modeled inference-tensor bytes,
6. measured process/device peak memory when available,
7. quality delta in **nats/token (or nats/character)** and **perplexity ratio**,
8. throughput/latency,
9. whether each number is measured, structurally modeled, or synthetic.

A GGUF Q4_K_M file size and LARC's internal row-Q4 estimator are different baselines and MUST NOT be interchanged.

The research target remains 10–30× lower peak resident inference memory than a named Q4-class deployment at the same context length while retaining useful capability.

## 2. Logical graph and physical storage

The manifest separates `logical_nodes` from `physical_bundles`.

A logical node may reference:

- `DENSE_REF`: ordinary quantized tensor page,
- `PROJECTION`: factorized `A(Bx)` / `A(U^T x)`,
- `RECURSIVE_REF`: repeated reference to one physical block bundle,
- `RELAXED_RECURSIVE_REF`: shared block plus depth-specific adapters/state,
- `PROJECTION_RESIDUAL`: projection core plus ordered residual pages.

Multiple logical layers MAY reference one physical page/bundle. Shared physical storage is counted once for residency. This is exact aliasing only when the model semantics genuinely share the parameter object; converting independent layers into a shared bundle is an approximation and requires separate quality accounting and provenance.

## 3. Canonical `Q4_ROW` v0.3 candidate

`Q4_ROW` is now defined identically across the Python and native reference paths:

- signed integer values: `q in [-8,7]`,
- stored nibble: `code = q + 8` in `[0,15]`,
- two codes per byte, low nibble first,
- zero code = `8`, used for odd-column padding,
- one IEEE-754 binary16 row scale,
- row scale:

`scale = max(max(row,0)/7, max(-row,0)/8, epsilon)`.

Encoding:

`q = clip(round(w / scale), -8, 7)`.

Decoding:

`w_hat = (code - 8) * scale`.

A standards-track release MUST include golden byte vectors and cross-language encoder/decoder conformance. The Run-3 reference golden vector is implemented in `tests/test_q4_format.py` and `tests/native_q4_smoke.cpp`.

## 4. Projection bundles

For operators `W_i` consuming a common input activation domain, LARC may store a basis and projected operators:

`y_i ~= A_i (B x)`.

### 4.1 Quantize-first fitting

The fitted projected operator MUST target the basis actually stored/executed, not an unavailable float precursor.

Reference fitting procedure:

1. derive calibration basis `B` from activations,
2. encode/decode it to obtain the actual stored `B_hat`,
3. compute latent calibration coordinates `Z = B_hat X`,
4. solve

`min_A ||W X - A Z||_F^2`,

with documented regularization,
5. encode `A`.

A direct-packed runtime MUST NOT materialize the reconstructed dense matrix `A B_hat` merely to execute the logical operator.

## 5. Progressive residual pages

Approximate cores may be refined by ordered pages prioritized by marginal validation gain per resident byte. Candidate residual codecs include sparse corrections, low-rank corrections, HRVQ64, and rotated/incoherent low-bit residuals.

### 5.1 Codebook accounting

For any vector/codebook codec, reported bpw MUST include the codebook's amortized storage unless the codebook is a normative globally shared object and the sharing scope is explicitly stated. For HRVQ64, a 256×64 FP16 codebook costs 32,768 B per stage; per-tensor codebooks cannot be omitted from compression accounting.

Quality tiers may be `CORE`, `R1`, `R2`, ..., `FULL_STORED`.

## 6. Latent KV cache

For each attention head or compatible head group, LARC may project historical K/V vectors into rank-r latent spaces:

`k_lat = B_k k`

`v_lat = B_v v`.

Historical full-dimensional K/V need not remain resident. Attention combines latent values and reconstructs only the weighted aggregate.

### 6.1 `LATENT_Q2_ROW`

Reference coefficient codec: asymmetric 2-bit per-token vectors with FP16 min and scale.

### 6.2 `LATENT_Q2_KIVI`

Research codec orientation:

- latent keys: asymmetric 2-bit per latent channel over token groups,
- latent values: asymmetric 2-bit per token,
- K/V bases: normally canonical Q4_ROW or Q8,
- partial current key group may remain higher precision or be incrementally repacked.

### 6.3 Quantized key-basis metric

A quantized/dequantized key basis `B_hat` is generally not row-orthonormal. Therefore plain latent dot products introduce a non-orthogonal metric.

The v0.3 reference stores an FP16 inverse-Gram correction:

`G_inv = (B_hat B_hat^T + lambda I)^-1`.

For `q_lat = q B_hat^T`, scoring uses:

`score_t ~= q_lat^T G_inv k_lat_t / sqrt(d_head)`.

Equivalent whitening/factorized metric representations are conforming if documented. The metric bytes MUST be included in residency accounting.

The manifest MUST record rank, head grouping, coefficient bit width, token-group size, basis codec, metric representation, and residual-tail policy.

## 7. Native compressed execution

A `DIRECT_PACKED` Q4 projected primitive computes:

1. `z = Bx` directly from packed B,
2. `y = Az` directly from packed A,

with intermediate scratch proportional to rank rather than dense matrix size.

Reference CPU source: `runtime/larc_q4.cpp`.
Reference CUDA/Triton source: `runtime/triton_q4.py`.

Dense FP16/FP32 reconstruction of an entire stored weight matrix is non-conforming for `DIRECT_PACKED`.

The current Triton source is a correctness/reference contract only until executed on CUDA hardware; its one-program-per-output-row GEMV layout is not a performance claim.

## 8. Paged file layout (v0.2 framing retained)

The implemented research container uses a 64-byte little-endian header (`<8sHHIQQQQQ8x`) and 64-byte page records (`<IHHQQQII24x`). Payload pages are aligned to a manifest-declared power-of-two alignment (reference default 4096 B).

Header fields: magic/version, flags, manifest length, page count, page-table offset, payload offset, exact file length.

Page fields: page ID, codec ID, flags, offset, stored length, logical length, CRC32, dependency group.

Current page flags include `REQUIRED`, `SHARED`, `REFINEMENT`, `STREAMABLE`, and `KV_BASIS`.

CRC32 currently covers payload pages individually; header/page-table authentication is not yet specified. A future standards-track release must define integrity policy (for example verify-on-open vs verify-on-touch) and protect page-table metadata.

## 9. Codec registry

| ID | Codec | Purpose | Status |
|---:|---|---|---|
| 0 | RAW | metadata/small tensors | implemented |
| 1 | Q4_ROW | canonical signed packed row-Q4 | v0.3 candidate implemented Python/C++ |
| 2 | Q8_ROW | signed row-Q8 | implemented reference |
| 3 | PROJECTION_Q4 | Q4 basis/projected factors | implemented research path |
| 4 | HRVQ64 | progressive vector residual | implemented; residual-only recommendation |
| 5 | LATENT_KV_BASIS_Q4 | low-bit latent-KV basis | implemented reference |
| 6 | SPARSE_RESCUE | sparse residual correction | reserved/planned |

IDs remain research-only until registry governance is frozen.

## 10. Memory-budget execution

A runtime SHOULD:

1. pin required/shared core pages,
2. allocate the selected KV representation at requested context,
3. reserve bounded kernel scratch,
4. admit refinement pages in validation-gain-per-byte order,
5. prefetch streamable pages before execution,
6. evict refinements before required pages,
7. report selected physical pages and unique resident payload bytes.

The current repository implements the storage primitives and accounting, not a complete production asynchronous page scheduler.

## 11. Conversion provenance

Preferred source formats are SafeTensors and GGUF. Converters SHOULD process tensor/shard-wise when possible.

Any activation-dependent conversion MUST record calibration provenance. Any conversion that changes cross-layer parameter sharing MUST record:

- original architecture,
- sharing map,
- initialization rule,
- recovery/distillation objective,
- recovery data provenance,
- optimizer/step count,
- equal-compute/smaller-model controls where reported,
- before/after quality on identical held-out data.

## 12. Validation levels

- **L0 Structural:** container/codec round trips and byte accounting.
- **L1 Operator:** compressed-domain execution correctness plus source/output error against explicit baselines.
- **L2 Conformance:** controlled trained autoregressive model exercising LARC representations.
- **L2C Post-training conversion:** independently parameterized controlled model trained first, then structurally converted/recovered, with same-token quality decomposition.
- **L3 External pretrained model:** independently hosted pretrained LLM conversion against a named deployment baseline and standard evaluation.
- **L4 Hardware:** measured process/device peak memory plus throughput on target CPU/GPU/accelerator.

L0/L1/L2/L2C results MUST NOT be promoted as L3/L4 evidence.

## 13. Current audited evidence boundary

Run 3 supersedes several Run-2 numerical claims. The strongest current controlled L2C result is:

- evaluation: 100,032 independently generated held-out characters, identical tokens for teacher/student paths,
- independent 16-block teacher NLL: 1.77359,
- recovered one-block shared student NLL before KV compression: 1.88556,
- compressed latent-Q2 / actual Q4-basis student NLL: 1.90953,
- structural conversion delta: +0.11197 nats/char, perplexity ×1.11848,
- KV compression delta: +0.02397 nats/char, perplexity ×1.02426,
- total delta: +0.13594 nats/char, perplexity ×1.14561,
- modeled same-context inference-tensor ratio: 10.6628×.

The memory result is **structural tensor accounting**, not measured RSS/VRAM. The evaluation corpus is a synthetic character-level template task, not a general LLM benchmark.

The old Run-2 post-training `+13.83% NLL` quality result is revoked because its teacher and compressed-student losses used different evaluation aggregation. The old native operator artifact reporting `0.046151` NMSE is also revoked because it does not reproduce from the checked-in source.

L3 remains unpassed because external checkpoint payloads were unavailable to this execution environment. L4 remains unpassed because no CUDA/Metal accelerator is available.
