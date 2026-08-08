# LARC v0.1 — Local Adaptive Representation & Compute

**Status:** experimental research specification  
**Extension:** `.larc`  
**Primary objective:** maximize retained model capability per resident byte on resource-constrained local hardware.

LARC is not intended to be a smaller wrapper around dense tensors. It is a runtime-oriented model representation in which a logical linear operator can be represented by a shared activation subspace, compressed projected operators, and independently loadable residual refinements.

## Design goals

1. Runtime-native compression: execute from compressed structures rather than reconstructing dense weights.
2. Bounded-memory inference: choose quality tiers under an explicit memory budget.
3. Progressive fidelity: refinements are independently addressable.
4. Shared structure: tensors with common activation spaces may share bases/codebooks/transforms.
5. Streaming conversion from sharded sources.
6. Architecture-neutral container graph.
7. Measurable degradation with calibration provenance.

## Core representation: projection bundles

For linear operators `W_i` consuming common activation `x`, store shared basis `U_k` and projected operators `A_i = W_i U_k`. Runtime computes `z = U_k^T x` once and `y_i_core = A_i z` for each operator.

The basis is learned from calibration activations rather than weight SVD alone. Candidate bundles include attention Q/K/V and MLP gate/up projections.

## Progressive residual representation

The core projection omits behavior outside the retained subspace. LARC therefore stores ordered residual refinement chunks. Prototype `HRVQ64` uses 64-weight vectors, one FP16 RMS scale per 256 weights, uint8 indexes into shared 256-entry additive codebooks, and progressive residual stages.

Nominal payload bitrates: HRVQ64-1 `0.1875 bpw`, HRVQ64-2 `0.3125 bpw`, HRVQ64-3 `0.4375 bpw`. These are storage targets, not quality claims.

## Quality tiers

- Tier 0 / CORE: projection bundle only.
- Tier 1 / RESIDUAL-A: highest-value residual pages.
- Tier 2 / RESIDUAL-B: additional residual stages/sparse corrections.
- Tier 3 / FULL-LOCAL: all stored refinements.

A runtime may choose tiers globally or per layer.

## Sensitive tensor fallback

The format permits ordinary quantized chunks for embeddings, normalization parameters, small tensors, highly sensitive layers, and output heads where structural compression is harmful.

## Prototype container

```text
8-byte magic/version  LARC\0\1\0\0
uint64 manifest length
UTF-8 JSON manifest
binary chunk 0
binary chunk 1
...
```

A production binary manifest/page table is deferred until codec requirements stabilize.

## Codec registry v0.1

| Codec | Purpose | Status |
|---|---|---|
| RAW | metadata/small payloads | implemented |
| Q4_ROW | row-wise signed 4-bit factors | implemented |
| Q8_ROW | row-wise signed 8-bit factors | implemented |
| PROJECTION_BUNDLE | shared activation basis + projected operators | implemented research object |
| HRVQ64 | progressive additive residual vector coding | implemented |
| SPARSE_RESCUE | high-impact residual entries/directions | planned |
| HADAMARD_ROTATED_* | incoherence preprocessing | planned |

## Runtime execution contract

1. Load/decode selected shared basis.
2. Compute `z = U^T x` once.
3. Compute projected outputs `A_i z`.
4. Add resident residual contributions without constructing dense `W_i`.
5. Evict pages according to memory budget.

Production kernels should fuse low-bit decode and GEMV/GEMM where practical.

## Compression accounting

LARC reports stored bytes, minimum resident bytes, selected-tier resident bytes, and peak scratch bytes. Compression claims must name their baseline, e.g. Q4_K_M GGUF.

## Validation

Track file size/bpw, peak RAM/VRAM, tokens/s, perplexity/KL, task benchmarks, held-out output error, and energy/token where possible. Primary derived metric: retained capability per resident GB.

## Non-goals v0.1

- bit-exact reconstruction of arbitrary dense source weights,
- claiming 10–30× quality-preserving compression before real-model evaluation,
- freezing a production ABI before codec validation,
- immediately replacing GGUF as interchange; initial tooling should import GGUF/SafeTensors and use LARC as an execution format.
