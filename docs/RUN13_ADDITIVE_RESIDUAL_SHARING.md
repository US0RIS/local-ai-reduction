# Run 13 — Full-rank shared base + low-rank layer residuals

## Hypothesis

For each like-shaped operator family across depth:

`W_layer ≈ B_group + U_layer V_layer^T`

where `B_group` is a **full-rank** physical matrix shared by several logical layers and `U_layer V_layer^T` is a small layer-specific additive residual.

This is deliberately different from both prior sharing failures:

- **Run 6:** logical layers were forced toward the same full block; Run 13 preserves a unique residual for every layer.
- **Run 7:** every logical matrix was represented through a shared low-rank output basis; Run 13 leaves the common component full-rank and only assumes that **inter-layer differences** have low effective rank.

If the residual-difference hypothesis is true, it can reduce effective parameter count by much more than same-parameter quantization can.

## Real-model scope

All 30 SmolLM2-135M decoder layers are included for:

- q_proj;
- k_proj;
- v_proj;
- o_proj;
- gate_proj;
- up_proj;
- down_proj.

This is the complete main decoder projection pool: 210 logical matrices.

Calibration uses WikiText-2 raw train activations. Evaluation uses disjoint WikiText-2 raw test activations.

- calibration rows/site: 128;
- evaluation rows/site: 64;
- two 512-token windows from each corpus.

## Grouping and ranks

The first diagnostic tests contiguous depth groups with:

- **2 physical bases** → 15 logical layers/base;
- **3 physical bases** → 10 logical layers/base;
- **6 physical bases** → 5 logical layers/base.

Each layer gets an activation-aware residual adapter at rank:

- 8;
- 16;
- 32.

These points were chosen because groups of 10–15 layers are required for structural reduction to become relevant to the original 10×-class objective, while rank8–16 adapters are small enough to preserve that economic advantage if they work.

## Activation-aware residual fitting

For logical layer `l` in a group:

1. compute the FP32 full-rank group mean `B`;
2. form exact residual `D_l = W_l - B`;
3. on calibration activations `X_l`, compute residual outputs `Y_l = X_l D_l^T`;
4. take the best rank-r SVD subspace of `Y_l`;
5. map that low-rank output subspace back to input space using a damped sample-space pseudoinverse of `X_l`.

The resulting factors are selected to reproduce the **functionally observed residual**, not merely the Frobenius-norm weight delta.

## Representation-matched packed diagnostic

The structural ceiling is evaluated first with FP32 bases/factors, but the actual component gate uses a packed representation:

- every physical shared base: `Q4_GROUP64`;
- every `U_layer`: `Q4_GROUP64`;
- every `V_layer^T`: `Q4_GROUP64`;
- no dense shadow weight is counted.

The baseline is an independent `Q4_GROUP64` copy of all 210 original projection matrices. Thus the reported projection-pool reduction is representation matched.

This is not claimed to be llama.cpp Q4_K_M parity. Run 9 independently establishes that external baseline.

## Precommitted component gate

Before observing results:

**Pass** only if at least one packed configuration simultaneously achieves:

- main-projection Q4 byte reduction **≥5×**; and
- held-out energy-weighted global operator NMSE **≤0.05**.

**Borderline** requires:

- reduction **≥4×**; and
- energy-weighted global NMSE **≤0.10**.

Anything else fails the component gate.

This is intentionally only an operator/component gate. Even a pass still requires end-to-end WikiText/task quality before a real model representation can be promoted.

## Why this matters to the headline target

Same-parameter W2 can at best provide roughly a 2× weight-payload reduction relative to a 4-bit representation before overhead. Additive residual sharing changes the **number of full matrices that must physically exist**.

For a group of `g` equal-shape matrices, rank `r`, output width `m`, and input width `n`, the unquantized structural parameter ratio is approximately:

`gmn / (mn + gr(m+n))`.

For large SmolLM2 MLP matrices, group10–15 with rank8–16 can theoretically remove several-fold more parameters than quantization alone. Run 13 determines whether the real inter-layer residuals actually permit that geometry.

## Evidence boundary

Run 13 is a real-pretrained operator diagnostic over the decoder projection pool. It does **not** include embeddings, normalization weights, the LM head, full-model perplexity, generation quality, a native fused residual-sharing kernel, process RSS, or VRAM. A component-gate pass is not a usable LARC model.

Execution note: the harness and gate above were merged to `main` before this execution-only documentation change was opened, so the measurement cannot alter its precommitted thresholds.
