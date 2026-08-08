# LARC v0.2 — Local Adaptive Representation & Compute

**Status:** experimental implementable specification  
**Extension:** `.larc`  
**Primary objective:** maximize retained model capability per **peak resident inference byte**, not merely per file byte.

LARC is a runtime-oriented representation for local neural language models. It deliberately permits a logical Transformer graph to differ from a list of independently stored dense tensors. Logical operators may reference shared physical parameter bundles, activation-subspace factors, depth adapters, progressive residual pages, and compressed KV-cache bases. A conforming runtime is expected to execute these representations without reconstructing the complete dense model.

## 1. Success metrics

Every compression claim MUST identify a baseline and report, separately:

1. complete file bytes,
2. unique resident weight bytes,
3. KV-cache bytes at a stated context length,
4. peak scratch/workspace bytes,
5. peak total resident inference bytes,
6. quality delta (NLL/perplexity plus task or generation metrics),
7. throughput/latency,
8. whether numbers are **measured**, **modeled from allocated structures**, or **synthetic**.

The initial research target is 10–30× lower peak resident inference memory than a Q4-class GGUF baseline at the same context length, while retaining useful model capability.

## 2. Logical graph and physical storage

The manifest separates `logical_nodes` from `physical_bundles`.

A logical linear node may be one of:

- `DENSE_REF`: ordinary quantized tensor page,
- `PROJECTION`: `A(Bx)` or `A(U^T x)` using shared basis pages,
- `RECURSIVE_REF`: repeated reference to one physical Transformer/block bundle,
- `RELAXED_RECURSIVE_REF`: shared block plus depth/recursion-specific adapter pages,
- `PROJECTION_RESIDUAL`: projection core plus ordered residual pages.

Multiple logical layers MAY reference one physical bundle. This is normative aliasing, not duplicated payload. A runtime MUST count a shared physical page once for residency.

### 2.1 Recursive/shared models

LARC-native models may reuse one physical block for multiple logical depths. Optional depth-specific low-rank adapters, norms, scalars, or routing metadata provide specialization while keeping the base bundle shared. This directly supports recursive/Universal-Transformer-like models and converted relaxed-recursive models.

## 3. Projection bundles

For operators `W_i` consuming a common input space, LARC may store a basis `B` and projected operators `A_i` such that:

`y_i ≈ A_i (B x)`.

The basis SHOULD be fit with calibration activations or another documented task-weighted objective rather than raw weight SVD alone. Attention Q/K/V and MLP gate/up are natural bundle candidates when their input domains are compatible.

A projection runtime MUST NOT allocate the reconstructed dense matrix `A_i B` merely to execute the operator.

## 4. Progressive residual pages

Approximate cores can be refined by ordered pages whose priority is defined by marginal validation gain per byte. Candidate residual codecs include:

- sparse high-impact corrections,
- low-rank error corrections,
- HRVQ64 additive vector pages,
- rotated/incoherent low-bit residuals.

Quality tiers:

- `CORE`: minimum executable representation,
- `R1`, `R2`, ...: progressively better residual sets,
- `FULL_STORED`: every refinement present in the file.

A memory-budgeted runtime MAY select different tiers per logical layer.

## 5. KV-cache representation

LARC v0.2 defines a **latent KV** execution class.

For each attention head or compatible head group, historical K/V vectors can be projected to learned rank-r bases:

`k_lat = B_k k`, `v_lat = B_v v`.

The cache stores the latent coefficients, not full historical K/V. Attention projects the current query into key-latent space, computes scores against latent keys, combines latent values, and reconstructs only the current weighted value aggregate.

### 5.1 `LATENT_Q2_ROW`

Reference codec: both latent K and V are asymmetric 2-bit per-token vectors with FP16 min/scale metadata.

### 5.2 `LATENT_Q2_KIVI`

Preferred research codec:

- latent keys: asymmetric 2-bit, per latent channel over token groups,
- latent values: asymmetric 2-bit, per token,
- K/V bases: normally Q4 or Q8,
- partial current key group MAY remain higher precision or be incrementally repacked.

The manifest MUST record rank, head grouping, token group size, coefficient bit width, basis codec, and residual-tail policy.

## 6. Native compressed execution

The canonical linear primitive is packed low-bit GEMV/GEMM. A conforming packed-Q4 projected kernel computes:

1. `z = Bx` directly from packed `B`,
2. `y = Az` directly from packed `A`,

with scratch proportional to projection rank, not `rows × columns`.

The reference CPU implementation is `runtime/larc_q4.cpp`; the CUDA/Triton reference contract is `runtime/triton_q4.py`.

Dense FP16/FP32 reconstruction of an entire stored weight matrix is non-conforming for the `DIRECT_PACKED` execution profile.

## 7. v0.2 paged file layout

The implemented research container uses fixed records and mmap-compatible page offsets.

### 7.1 Header

64 bytes, little-endian (`<8sHHIQQQQQ8x`):

| Field | Type | Meaning |
|---|---|---|
| magic | 8 bytes | `LARCv2\0\0` |
| major/minor | u16/u16 | format version |
| flags | u32 | file-level flags |
| manifest_length | u64 | UTF-8 JSON manifest bytes after header |
| page_count | u64 | fixed page-record count |
| page_table_offset | u64 | offset of page table |
| data_offset | u64 | first payload region |
| file_length | u64 | exact file length |

### 7.2 Page record

Each page record is 64 bytes (`<IHHQQQII24x`):

- `page_id: u32`
- `codec_id: u16`
- `flags: u16`
- `offset: u64`
- `stored_length: u64`
- `logical_length: u64`
- `crc32: u32`
- `dependency_group: u32`

Payload pages are aligned to the manifest's power-of-two alignment (reference default 4096 bytes). CRC32 covers stored payload bytes.

### 7.3 Page flags

- `REQUIRED`
- `SHARED`
- `REFINEMENT`
- `STREAMABLE`
- `KV_BASIS`

A runtime MAY mmap the file and expose page views without copying. The same `SHARED` page referenced by multiple logical nodes counts once in resident-payload accounting.

## 8. Codec registry v0.2

| ID | Codec | Purpose | Status |
|---:|---|---|---|
| 0 | RAW | metadata/small tensors | implemented |
| 1 | Q4_ROW | signed packed row-Q4 | implemented |
| 2 | Q8_ROW | signed row-Q8 | implemented |
| 3 | PROJECTION_Q4 | Q4 basis/projected factors | implemented |
| 4 | HRVQ64 | progressive vector residual | implemented, residual-only recommendation |
| 5 | LATENT_KV_BASIS_Q4 | latent-KV basis payload | implemented reference |
| 6 | SPARSE_RESCUE | sparse residual correction | reserved/planned |

Codec IDs are stable only within v0.2 research files; a future standards-track release will define registry governance.

## 9. Memory-budget execution

A runtime accepts a resident-memory budget and SHOULD:

1. pin required/shared core pages,
2. allocate KV cache according to requested context and KV tier,
3. reserve bounded kernel scratch,
4. admit residual pages in priority order while under budget,
5. prefetch streamable pages before their logical node executes,
6. evict refinements before required core pages.

The runtime MUST be able to report the selected physical pages and their unique stored/resident byte total.

## 10. Sensitive fallbacks

Embeddings, output heads, normalization tensors, very small operators, or empirically sensitive layers may use ordinary quantized pages. LARC does not require structural compression where it worsens quality-per-byte.

## 11. Conversion

Preferred source formats are SafeTensors and GGUF. Converters SHOULD operate shard-by-shard or tensor-by-tensor so a user does not need two full dense model copies locally. Calibration data and converter settings MUST be recorded in provenance metadata when a representation depends on activations.

## 12. Validation levels

- **L0 Structural:** container/codec round trips and byte accounting.
- **L1 Operator:** held-out operator error plus compressed-domain kernel correctness.
- **L2 Conformance model:** trained autoregressive model; file/weight/KV/total-memory and NLL gates measured from executable representations.
- **L3 External pretrained model:** independent pretrained LLM converted after training; same-context quality and total-memory comparison against named GGUF baseline.
- **L4 Hardware:** measured CPU/GPU/accelerator peak memory and throughput on target hardware.

A project MUST NOT promote L0/L1 modeled results as L3/L4 evidence.

## 13. Current implementation boundary

v0.2 has implemented L0, L1, and an L2 recurrent conformance path. The repository contains an L3 SmolLM2-135M harness, but external checkpoint retrieval / hosted runner availability has prevented completion of that benchmark in the current execution environment. CPU packed-domain kernels are measured locally; the Triton GPU kernel is source-complete but not hardware-validated here.
