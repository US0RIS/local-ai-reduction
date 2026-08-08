# Run 15 — Factorize the tied token embedding / LM head

## Why this component is mandatory

SmolLM2-135M uses one tied vocabulary matrix for input embeddings and the output language-model head. With vocabulary 49,152 and hidden width 576, that matrix contains **28,311,552 parameters**. Relative to the model's 134,515,008 unique parameters, it is approximately **21.05%** of the model.

Therefore leaving the tied vocabulary matrix at the Q4 baseline creates a hard weight-storage floor: even if every other model parameter became free, maximum weight-only reduction would be only about **4.75×**.

A 10× Q4-relative weight goal therefore requires substantial compression of this matrix as well as the decoder.

## Representation

Let the tied vocabulary matrix be `E ∈ R^(V×D)`.

Run 15 computes the optimal Frobenius rank-r right subspace from the small Gram matrix `E^T E` and writes:

`E ≈ A B`

with:

- `A ∈ R^(V×r)`;
- `B ∈ R^(r×D)`.

The same factors implement both sides of the tied matrix:

- input embedding: `embedding(token) = A[token] B`;
- output head: `logits(h) = (h B^T) A^T`.

This preserves weight tying at the representation level; there are not separate input and output factor sets.

Tested ranks: **64, 128, 192, 256**.

## Packed byte contract

The deployment diagnostic uses:

- `A`: `Q4_GROUP64`;
- `B`: `Q4_GROUP64`;
- no dense `E` shadow.

The component baseline is one tied `E` stored as `Q4_GROUP64`.

The artifact also records the FP16 structural factor ratio, but component promotion uses the packed Q4 factor result.

## Three quality tests

All factors are fitted from the pretrained vocabulary matrix only. No evaluation activations or labels are used to fit them.

The fixed evaluation slice is the first 8,192 prediction tokens of WikiText-2 raw test at context512.

### 1. Input embedding fidelity

For every token occurrence in the held-out stream:

`NMSE = ||A[token]B - E[token]||² / ||E[token]||²`

summed over occurrences. This weights vocabulary rows by actual held-out token frequency rather than treating rare and common tokens identically.

### 2. Head-only next-token quality

The original decoder and original input embeddings produce final hidden states `h`. Only the output head is replaced by the factorized head. The artifact records NLL/PPL ratio versus the original tied head.

This isolates output-head loss.

### 3. Integrated input + head quality

Input token IDs are embedded using the factors, those factorized embeddings propagate through the **entire untouched 30-layer decoder**, and final logits are also produced through the same factors.

This is the decisive component test because input embedding error is allowed to affect every decoder layer before output scoring.

It is still not a full LARC candidate: decoder weights are unchanged and the evaluation is a fixed slice rather than the final full-corpus standard benchmark.

## Precommitted component gate

**Pass** requires one packed Q4 factorization to satisfy all of:

- tied-matrix reduction ≥ **4×** versus tied Q4_GROUP64 `E`;
- occurrence-weighted input-embedding NMSE ≤ **0.05**;
- head-only PPL ratio ≤ **1.05×** original;
- integrated factorized-input+head PPL ratio ≤ **1.10×** original.

**Borderline** requires:

- reduction ≥ **3×**;
- embedding NMSE ≤ **0.10**;
- head-only PPL ratio ≤ **1.10×**;
- integrated ratio ≤ **1.25×**.

These thresholds are fixed before execution.

## Interpretation

A pass would remove the tied vocabulary matrix as a fundamental obstacle to a 10× whole-model weight target and would justify full-corpus validation plus a native two-stage embedding/head kernel.

A failure at rank128 or higher would mean the vocabulary matrix itself requires enough independent capacity to become a major lower bound on this model's achievable compression. In that case, a new architecture with a factorized embedding parameterization **trained from the beginning** becomes more relevant than post-hoc SVD.

## Evidence boundary

Run 15 is a real-pretrained **component** diagnostic. The decoder remains original. Q4_GROUP64 is the component byte reference, not llama.cpp Q4_K_M. The fixed 8K-token slice is not promoted as full WikiText-2 quality. There is no native packed factor kernel, process RSS, VRAM, or whole-model compression claim.

Execution provenance: the implementation, rank set, fixed evaluation slice, Q4 factor byte contract, and pass/borderline thresholds above were merged to `main` at `971dc36723b3e2c24b3f56bb966f61f157923ab3` before this execution-only documentation update was opened.
