# Run 17 — Direct-packed tied-vocabulary product quantization

## Motivation

Run 12's exact Q4_K_M file bound gives the entire 10× candidate only **10.545 MB**. SmolLM2's single tied 49,152×576 input-embedding / LM-head matrix occupies **15.0405 MB** even under the project's Q4_GROUP64 component arithmetic. Therefore the vocabulary interface alone already exceeds the complete 10× file budget.

Run 15 tests global low-rank factorization of that matrix. Its initial low-rank result is severe and is being independently validated with a full-rank control. Run 17 tests a structurally different representation that preserves full-rank geometry: **product-quantized token rows**.

## Representation

Each pretrained token vector is normalized by one per-token scalar and split into fixed contiguous subspaces. For token `t`:

`E[t] ≈ norm[t] × concat(C_0[code[t,0]], …, C_(M-1)[code[t,M-1]])`.

Storage:

- one FP16 norm per vocabulary row;
- one uint8 centroid code per token/subspace;
- 256 FP16 centroids per subspace;
- no dense vocabulary matrix.

Tested subspace dimensions: **8, 12, 16, 24, 32**. Hidden width 576 is divisible by all five.

Every subspace codebook is fitted using six deterministic full-vocabulary Lloyd iterations on the pretrained embedding weights only. No WikiText activation, token-frequency, label, or quality data enters the fit.

## Exact byte economics

Because `M × subdim = 576`, total FP16 codebook storage is constant across variants:

- codebooks: `576 × 256 × 2 = 294,912 B`;
- token norms: `49,152 × 2 = 98,304 B`;
- token codes: `49,152 × (576/subdim) B`.

Against the 15,040,512-byte tied Q4_GROUP64 matrix, approximate component reductions are therefore:

- subdim8: ~3.8×;
- subdim12: ~5.5×;
- subdim16: ~7.0×;
- subdim24: ~9.6×;
- subdim32: ~11.7×.

The exact artifact recomputes these values from the loaded model.

## Direct packed inference semantics

### Input lookup

For one token, gather one centroid from each subspace, concatenate the centroid subvectors, and multiply by the token's FP16 norm.

### Output head

For hidden vector `h`, split `h` into the same subspaces. For each subspace compute only 256 centroid dot products:

`table_s[k] = dot(h_s, C_s[k])`.

Then for token `t`:

`logit[t] = norm[t] × Σ_s table_s[code[t,s]]`.

Thus a dense 49,152×576 output matrix never needs to exist. The reference harness checks this direct packed math against a separately decoded dense matrix on random hidden vectors; semantic maximum absolute error must remain ≤1e-4 for any gated result.

## Quality evaluation

Fixed leading WikiText-2 raw test slice:

- 8,192 next-token predictions;
- context512;
- original SmolLM2 decoder.

For every PQ point:

1. whole-vocabulary weight NMSE;
2. held-out occurrence-weighted input-embedding NMSE;
3. head-only next-token PPL using original decoder hidden states;
4. integrated factorized input lookup propagated through the original 30-layer decoder plus direct packed output head;
5. direct-packed semantic conformance error.

The packed representation is evaluated with codebooks and token norms rounded to FP16, matching byte accounting.

## Precommitted component gate

**Pass** requires all of:

- tied-matrix reduction ≥5× vs Q4_GROUP64;
- held-out occurrence-weighted embedding NMSE ≤0.05;
- head-only PPL ratio ≤1.05× original;
- integrated input+head PPL ratio ≤1.10×;
- direct packed semantic max error ≤1e-4.

**Borderline** requires:

- reduction ≥4×;
- occurrence NMSE ≤0.10;
- head-only PPL ratio ≤1.15×;
- integrated ratio ≤1.50×;
- semantic error ≤1e-4.

## Decision value

A pass would provide a representation that attacks the vocabulary floor with genuinely sub-bit average information per original embedding weight while retaining a direct compressed-domain input/head algorithm. It would then need a native kernel and full-corpus validation.

A failure would show that post-hoc vocabulary compression also needs learned end-to-end adaptation; the remaining path would be a tokenizer/vocabulary interface trained jointly with the recursive low-description-length model rather than another standalone embedding codec.

## Evidence boundary

Run 17 is a real-pretrained tied-vocabulary component diagnostic. Q4_GROUP64 is the component reference, not llama.cpp Q4_K_M. The fixed 8K-token slice is not a final standard benchmark. No native optimized product-quantized softmax kernel, process RSS, VRAM, or total-model LARC result is claimed.
