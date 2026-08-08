# LARC v0.4-candidate — Local Adaptive Representation & Compute

**Status:** experimental research specification; v0.2 container framing retained  
**Extension:** `.larc`  
**Primary objective:** maximize retained model capability per **peak resident inference byte**, not merely per file byte.

LARC is a runtime-oriented representation in which the logical Transformer graph can differ from a list of independently stored dense tensors. Logical operators may reference shared physical parameter bundles, activation-subspace factors, depth adapters, progressive residual pages, and compressed KV-cache bases. A conforming direct-packed runtime executes these representations without reconstructing the complete dense model.

## 1. Required reporting semantics

Every promoted compression claim MUST identify its exact baseline and report separately:

1. complete serialized file bytes,
2. unique resident weight bytes,
3. KV-cache bytes at a stated context length,
4. scratch/workspace bytes at that context,
5. modeled inference-tensor bytes,
6. measured process/device peak memory when available,
7. quality delta in **nats/token (or nats/character)** and **perplexity ratio**,
8. throughput/latency,
9. whether each number is measured, structurally modeled, or synthetic,
10. training/evaluation seed distribution for controlled learned-model claims.

Quality MUST execute the representation whose bytes are charged. A GGUF Q4_K_M file/runtime, LARC's row-Q4 reference, and an FP32 teacher are different baselines and MUST NOT be interchanged.

The research target remains 10–30× lower peak resident inference memory than a named competitive Q4-class deployment at the same context length while retaining useful capability.

## 2. Logical graph and physical storage

The manifest separates `logical_nodes` from `physical_bundles`.

A logical node may reference:

- `DENSE_REF`: ordinary quantized tensor page,
- `PROJECTION`: factorized `A(Bx)` / `A(U^T x)`,
- `RECURSIVE_REF`: repeated reference to one physical block bundle,
- `RELAXED_RECURSIVE_REF`: shared block plus depth-specific adapters/state,
- `PROJECTION_RESIDUAL`: projection core plus ordered residual pages.

Multiple logical layers MAY reference one physical page/bundle. Shared physical storage is counted once for residency. This is exact aliasing only when the model semantics genuinely share the parameter object; converting independent layers into a shared bundle is an approximation and requires separate quality accounting and provenance.

KV coefficient histories remain logically distinct per logical attention invocation even when those invocations reference one physical block. Conversely, K/V projection bases and basis metrics MAY be physical shared objects. Memory accounting MUST therefore distinguish `logical_kv_histories` from `physical_basis_sets`.

## 3. Canonical signed Q4 family

### 3.1 `Q4_ROW`

Canonical nibble semantics:

- signed values `q in [-8,7]`,
- stored nibble `code=q+8`,
- two codes/byte, low nibble first,
- zero/padding code `8`,
- IEEE-754 binary16 scale,
- scale for group `g`:

`scale = max(max(g,0)/7, max(-g,0)/8, epsilon)`.

Encoding: `q=clip(round(w/scale),-8,7)`.
Decoding: `w_hat=(code-8)*scale`.

`Q4_ROW` uses one scale for the complete row.

### 3.2 `Q4_GROUP64`

Run-5 research codec. Each matrix row is partitioned into contiguous groups of at most 64 weights and the same signed-Q4 rule is applied independently to every group. Each group stores one FP16 scale.

The controlled recurrent model uses this codec because shared-block rows exhibit materially larger absmax/RMS than independent teacher rows, making full-row scaling inefficient.

A standards-track release MUST include golden byte vectors and cross-language conformance for every frozen group size.

## 4. Projection bundles

For operators `W_i` consuming a common input activation domain, LARC may store a basis and projected operators:

`y_i ~= A_i (B x)`.

### 4.1 Quantize-first fitting

The fitted projected operator MUST target the basis actually stored/executed:

1. derive calibration basis `B`,
2. encode/decode it to actual `B_hat`,
3. compute `Z=B_hat X`,
4. solve `min_A ||W X - A Z||_F^2` with documented regularization,
5. encode `A`.

A direct-packed runtime MUST NOT materialize the dense matrix `A B_hat` merely to execute the logical operator.

## 5. Progressive residual pages

Approximate cores may be refined by ordered pages prioritized by marginal validation gain per resident byte. Candidate residuals include sparse corrections, low-rank corrections, HRVQ64, and rotated/incoherent low-bit residuals.

For any vector/codebook codec, reported bpw MUST include amortized codebook storage unless a normative globally shared object is explicitly identified. For HRVQ64, one 256×64 FP16 codebook is 32,768 B per stage.

## 6. Latent KV cache

For each compatible attention head:

`k_lat = B_k k`, `v_lat = B_v v`.

Historical full-dimensional K/V need not remain resident. Attention operates on latent history and reconstructs only the weighted V aggregate.

### 6.1 Quantized basis metrics

A decoded low-bit basis `B_hat` is generally not row-orthonormal. The v0.4 reference uses the same ridge-stabilized inverse Gram for K and V:

`G_inv = (B_hat B_hat^T + lambda I)^-1`,

where the controlled reference uses

`lambda = 1e-5 * mean(diag(B_hat B_hat^T))`

per head.

Key score:

`score_t ~= q_lat^T G_k^-1 k_lat_t / sqrt(d_head)`.

Value reconstruction:

`v_full ~= v_lat G_v^-1 B_v_hat`.

Equivalent whitening/factorized forms are conforming if their byte accounting and numerical semantics are documented.

### 6.2 `LATENT_Q2_ROW`

Asymmetric 2-bit coefficients with one FP16 minimum and FP16 scale per token vector.

### 6.3 `LATENT_Q2_KIVI`

Research orientation: latent keys quantized per channel over token groups and values per token. This remains useful structural prior art but is not the selected Run-5 controlled codec.

### 6.4 `LATENT_Q2_GROUP_SCALAR`

Run-5 controlled codec:

- Q2 latent coefficients,
- one FP16 minimum + FP16 scale shared across both token and latent dimensions for each group,
- group size is explicit in the manifest and is a rate-distortion parameter,
- K and V may use independent group sizes,
- an incomplete causal group MAY be retained as a higher-precision residual tail; its precision and maximum resident bytes MUST be charged,
- Run-5 selected controlled profile uses group size **3** for both K and V and an FP16 residual tail.

For recurrent/shared models, `logical_kv_histories` and `physical_basis_sets` MUST be encoded separately. Run-5 controlled geometry has 16 logical histories but one physical K/V basis set.

The manifest MUST record rank, logical history count, physical basis-set count, head grouping, coefficient bits, token-group size, basis codec, basis scale codec, metric representation/ridge, and residual-tail policy.

## 7. Native compressed execution

A `DIRECT_PACKED` projected primitive computes `z=Bx` and `y=Az` directly from packed factors with rank-sized intermediate scratch. Dense reconstruction of an entire stored weight matrix is non-conforming for `DIRECT_PACKED`.

Reference CPU: `runtime/larc_q4.cpp`.  
Reference CUDA/Triton: `runtime/triton_q4.py`.

The Triton source remains a reference contract until executed and benchmarked on actual CUDA hardware.

## 8. Paged file layout

The implemented research container retains v0.2 framing:

- 64-byte little-endian header `<8sHHIQQQQQ8x`,
- 64-byte page entries `<IHHQQQII24x`,
- manifest-declared power-of-two payload alignment, reference 4096 B,
- per-page CRC32,
- dependency groups,
- `REQUIRED`, `SHARED`, `REFINEMENT`, `STREAMABLE`, `KV_BASIS` flags.

Header/page-table authentication and production integrity policy remain open.

## 9. Research codec registry

| ID | Codec | Purpose | Status |
|---:|---|---|---|
| 0 | RAW | metadata/small tensors | implemented |
| 1 | Q4_ROW | canonical signed row-Q4 | implemented Python/C++ |
| 2 | Q8_ROW | signed row-Q8 | implemented reference |
| 3 | PROJECTION_Q4 | low-rank Q4 factors | research path |
| 4 | HRVQ64 | residual/refinement | implemented; residual-only recommendation |
| 5 | LATENT_KV_BASIS_Q4 | low-bit K/V basis | implemented reference |
| 6 | SPARSE_RESCUE | sparse residual | reserved |
| 7 | Q4_GROUP64 | signed Q4 with FP16 scale / <=64 weights | Run-5 controlled candidate |
| 8 | LATENT_Q2_GROUP_SCALAR | grouped scalar metadata latent Q2 | Run-5 controlled candidate |

IDs are not standards-stable until registry governance is frozen.

## 10. Memory-budget execution

A runtime SHOULD pin required/shared core pages, allocate KV at requested context, reserve context-dependent kernel scratch, admit refinements by validation gain/byte, prefetch streamable pages, evict refinements before required pages, and report unique resident physical payloads.

Reference workspace formulas are accounting models, not substitutes for measured allocator/process/device peaks.

## 11. Conversion provenance

Any conversion changing cross-layer sharing MUST record:

- original architecture,
- sharing map,
- initialization rule,
- function-prefit/distillation/recovery objective,
- recovery data provenance,
- optimizer/schedule/steps,
- exact quantization projection schedule,
- calibration data for activation/KV bases,
- seed list,
- quality on identical held-out data,
- smaller-model/equal-compute controls where reported.

Run-5 controlled conversion uses teacher-layer function prefit followed by hard-projected group-64 QAT recovery.

## 12. Validation levels

- **L0 Structural:** container/codec round trips and byte accounting.
- **L1 Operator:** compressed-domain execution plus explicit source/output error.
- **L2 Conformance:** controlled trained autoregressive model.
- **L2C Post-training conversion:** independently parameterized controlled model trained first, then converted/recovered.
- **L3 External pretrained model:** independently hosted pretrained LLM against a named competitive deployment baseline and standard evaluation.
- **L4 Hardware:** measured process/device peak memory plus throughput.

L0/L1/L2/L2C MUST NOT be promoted as L3/L4 evidence.

## 13. Current audited evidence boundary — Run 5

Selected controlled candidate:

- 16-independent-block teacher trained first;
- one recurrent physical block after conversion;
- 80-step teacher-layer function prefit;
- 200-step hard-projected group-64 QAT recovery;
- group-64 Q4 weights;
- rank-16 grouped latent-Q2 KV with 3-token K/V groups;
- shared Q4 K/V bases;
- FP16 inverse-Grams for K and V;
- FP16 incomplete-group tail;
- context-dependent reference workspace.

Baseline is **the project's simple row-Q4 teacher + FP16 KV**, not Q4_K_M.

Modeled total-memory ratio:

- context 64: **11.297×**,
- context 512: **10.986×**,
- context 2K: **10.887×**,
- context 8K: **10.857×**.

Five training seeds, 100,032 held-out characters each:

- mean delta vs same row-Q4 baseline: **+0.03551 nats/char**,
- sample std: **0.16078**,
- mean perplexity ratio: **1.04705×**,
- ratio range: **0.8969×–1.2363×**.

Absolute mean perplexity ratio vs FP32 teacher: **1.37724×**.

Thus the controlled ≥10× gate is passed only at **L2C versus the project row-Q4 reference**. This is not evidence of Q4_K_M parity, not a real pretrained-model result, and not measured RAM/VRAM.

L3 remains open because an accessible external checkpoint payload has not been available to the execution environment. The decisive transfer test is real Transformer activation/rank geometry plus an independent 135M+ conversion. L4 remains open because no target CUDA/Metal hardware measurement has been performed.
