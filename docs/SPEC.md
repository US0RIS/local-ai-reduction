# LARC v0.3-candidate — Local Adaptive Representation & Compute

**Status:** experimental research specification; v0.2 container framing retained  
**Extension:** `.larc`  
**Current research target:** **10×** versus a named Q4-class deployment at the same context, while retaining useful capability.

LARC permits the logical Transformer graph to differ from a list of independently stored dense tensors. A `DIRECT_PACKED` runtime executes stored representations without reconstructing the complete dense model.

## 1. Mandatory claim semantics

Every memory/compression claim MUST state:

1. exact named baseline and codec,
2. context length and batch/generation mode,
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

**Representation consistency:** if a memory claim charges a stored low-bit representation, the associated quality path MUST execute those stored/dequantized values or their direct-packed mathematical equivalent.

**10× completeness rule:** a claim of “10×” MUST cover the complete byte pool named by the claim. Dominant-matrix-only or weights-only ratios MUST be labeled as such and cannot be promoted as complete-model/total-memory 10× evidence.

## 2. Logical graph / physical bundles

The manifest separates logical nodes from physical bundles. Research node types include:

- `DENSE_REF`
- `PROJECTION`
- `RECURSIVE_REF`
- `RELAXED_RECURSIVE_REF`
- `PROJECTION_RESIDUAL`
- **`SHARED_RESIDUAL_REF`**

A shared physical page is counted once. Exact aliasing is lossless only when the logical model genuinely reuses that parameter object. Collapsing independent pretrained layers into shared state is lossy conversion and requires quality/recovery provenance.

### 2.1 `SHARED_RESIDUAL_REF` / SoftShare

Run-5 primary weight form:

`W_(layer,type) = S_type + A_(layer,type) B_(layer,type)`.

The physical bundle contains:

- one shared full-rank base `S_type`, normally `Q4_ROW`;
- one depth-specific low-rank A/B pair per logical layer;
- depth-specific small state where applicable.

The manifest MUST record the shared-base page, logical layer, matrix type, residual rank, factor codec, and dependency group.

A conforming direct-packed runtime evaluates:

`y = Sx + A(Bx)`

without constructing `W = S + AB`.

Run-5 policy is **budget-first rather than compression-maximal**: ranks/rescue pages SHOULD be increased while validation gain is positive until the configured 10× deployment boundary is approached. Extra compression is not itself an objective.

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

Golden tests exercise both signed endpoints.

## 4. Projection and residual fitting

For projected operators, fitting MUST target the representation actually stored/executed.

For SoftShare conversion this means:

1. derive/accumulate float shared base `S`,
2. encode/decode it to obtain stored `S_hat`,
3. discard the unavailable float precursor for quality/reference purposes,
4. fit each residual against `R_layer = W_layer - S_hat`,
5. encode residual A/B factors,
6. if recovery training is used, constrain/reproject parameters to the storage grid for representation-consistent quality claims.

This prevents quality from benefiting from an unstored float shared base.

## 5. Progressive rescue pages

Residual/refinement pages may contain additional low-rank capacity, sparse corrections, HRVQ64, or rotated low-bit corrections.

For a fixed deployment target such as 10×, refinements SHOULD be ordered by **held-out validation gain per resident byte**. A converter/runtime may add rank increments or rescue pages until the byte ceiling is reached.

Any codebook bpw claim MUST include amortized codebook bytes unless codebook sharing scope is normative and explicit.

## 6. Latent KV

Historical K/V may be represented in rank-r latent coordinates:

`k_lat = B_k k`, `v_lat = B_v v`.

### 6.1 Q2 coefficients and metadata

`LATENT_Q2_ROW` stores asymmetric 2-bit coefficients. Supported research metadata profiles:

- FP16 min + FP16 scale: 4 metadata bytes/vector;
- E4M3-FN min + E4M3-FN scale: 2 metadata bytes/vector.

Metadata format MUST be declared. `LATENT_Q2_KIVI` may group key metadata while retaining per-token value metadata.

### 6.2 Quantized-basis metrics

For stored/dequantized basis `B_hat`, both K and V require inverse-Gram correction unless an equivalent transform is folded elsewhere:

`G_K^-1 = (B_K_hat B_K_hat^T + lambda I)^-1`

`G_V^-1 = (B_V_hat B_V_hat^T + lambda I)^-1`.

Scores use `G_K^-1`; value reconstruction uses `G_V^-1`. Metric bytes and all Q4 basis scales MUST be charged.

### 6.3 Rank policy

KV rank is not minimized independently. Under a fixed 10× deployment target, KV rank SHOULD be increased when doing so improves quality and the complete deployment budget remains compliant.

## 7. Direct packed execution

### 7.1 Q4 / SoftShare

`runtime/larc_q4.cpp` implements:

- `q4_gemv`
- `q4_gemv_add`
- `q4_transposed_gemv`
- projected `A(Bx)`
- `q4_shared_residual_gemv` for `Sx + A(Bx)`.

SoftShare scratch is rank-sized; no full per-layer dense matrix is materialized.

### 7.2 Packed latent-Q2 attention

`runtime/larc_q2_attention.cpp` consumes packed Q2 historical K/V, E4M3 metadata, Q4 bases, and FP16 K/V inverse-Gram matrices without materializing FP32 historical `T×r` arrays.

Reference scratch is `T + 4r` FP32 values for one head and may be reused across heads. This is a correctness/memory contract, not an optimized throughput claim.

CUDA/Triton code remains reference-only until hardware validation.

## 8. Paged file layout and streaming output

The v0.2 framing remains: 64-byte header, 64-byte page records, aligned payloads, CRC32, dependency groups, and `REQUIRED/SHARED/REFINEMENT/STREAMABLE/KV_BASIS` flags.

`LARCv2StreamWriter` is the required conversion pattern for large outputs when bounded conversion residency is claimed:

1. page count/manifest are declared;
2. page-table space is reserved;
3. each compressed page is written immediately;
4. only page records are retained;
5. table/header are finalized after the last payload.

A converter claiming bounded output residency MUST NOT accumulate all payload bytes before writing.

## 9. Source streaming / SafeTensors range access

A bounded-source-residency converter MAY read a sharded SafeTensors source by tensor byte range.

For remote HTTP sources:

- exact byte ranges SHOULD be requested;
- a response MUST provide HTTP `206` and `Content-Range` before being accepted as a range response;
- a server that ignores Range MUST be rejected rather than silently consumed as a full multi-gigabyte shard.

The source index, tensor names/ranges, and whether complete local source files were required MUST be recorded in conversion provenance.

A two-pass SoftShare converter may reread source tensors; bounded **residency** does not imply minimum network traffic.

## 10. Memory-budget execution

A runtime SHOULD pin required/shared pages, allocate the selected KV tier at requested context, reserve bounded scratch, admit refinement pages by marginal validation gain/byte, and report unique resident pages/bytes.

A total-memory claim at one context MUST NOT be generalized to another context without a context sweep or measurement.

For a fixed 10× policy, the runtime/converter SHOULD spend unused byte headroom on quality-improving capacity instead of reporting gratuitously higher compression.

## 11. Conversion / evaluation provenance

Record source architecture, sharing map, rank allocation, initialization, recovery objective/data, optimizer schedule, steps, calibration data, checkpoint-selection data, and final evaluation data.

Training, checkpoint selection, compression calibration, and final evaluation SHOULD be disjoint for promoted held-out results.

## 12. Artifact provenance

`benchmarks/INDEX.json` is authoritative about promoted/historical evidence. Promoted artifacts MUST name a committed generator and pass `tools/check_benchmark_provenance.py`.

A numerical artifact that cannot be reproduced from its declared generator MUST be revoked/demoted.

Exact external baseline bytes SHOULD replace rounded UI sizes whenever an exact file size/LFS pointer is available.

## 13. Validation levels

- **L0:** format/codec integrity and byte accounting.
- **L1:** operator/runtime correctness and explicit source/output error.
- **L2:** controlled trained conformance model.
- **L2C:** post-training conversion of an independently parameterized controlled model, with representation-consistent quality.
- **L3:** independently hosted pretrained LLM against a named deployment baseline and standard evaluation.
- **L4:** measured CPU/GPU/accelerator peak memory and throughput.

L0–L2C MUST NOT be presented as L3/L4.

## 14. Current research boundary

Run 4 hard recursion remains an important reference result but is no longer the preferred 10× architecture.

Authoritative Run-5 controlled strategy selection uses **canonical Q4_ROW for the residual factors, matching the converter/runtime**:

- Q4 teacher NLL: **1.90547**;
- SoftShare rank3: NLL **1.85275**, ppl×**0.94864**, complete tiny-model tensor reduction **7.099×**;
- rank2: NLL **1.98593**, ppl×**1.08378**, reduction **8.095×**;
- rank1: NLL **1.91066**, ppl×**1.00520**, reduction **8.411×**.

Earlier Run-5 toy results that paired grouped-Q4 residual-factor quality with a different converter/runtime codec are revoked. The tiny d=128 model is not a useful complete-file 10× proxy because Q4 per-row scale bytes dominate 1–3-wide factors.

Named real planning baseline:

- Mistral-7B-v0.1 Q4_K_M exact file bytes: **4,368,438,912**;
- exact 10× file ceiling: **436,843,891 B**;
- rank96 weight `.larc` estimate: **371,302,608 B** before auxiliary resources;
- with conservative 4 MiB tokenizer/config reserve: **375,496,912 B = 11.63375×**;
- remaining complete-file planning headroom: **61,346,979 B**;
- rank96/KV64 tensor model: **11.8600×** for weights + 4K KV;
- equal-common-scratch headroom before tensor ratio reaches 10×: **85,478,016 B**.

Native packed SoftShare execution is L1-validated and a local synthetic two-shard streaming conversion passes. Real Mistral conversion/quality (L3) and measured deployment memory (L4) remain open.
