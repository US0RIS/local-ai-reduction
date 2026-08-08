# Run 6 — real-pretrained-model falsification

## Objective

Move LARC from controlled synthetic L2C evidence toward L3 by testing the two assumptions that most directly determine whether the current architecture can transfer to a real pretrained Transformer:

1. **Activation/operator low-rank structure:** can real SmolLM2 projection operators be represented as `A(Bx)` at ranks aggressive enough to matter, on held-out activations?
2. **Cross-depth sharing:** can independently trained real decoder layers be physically shared, and can a small real layer group recover after exact aliasing when the stored weight representation is enforced during optimization?

Target checkpoint: `HuggingFaceTB/SmolLM2-135M`, Apache-2.0, 30 decoder layers, hidden size 576, FFN 1536, 9 attention heads, 3 KV heads, max context 8192.

## A. Real activation/operator spectra

Representative layers: `0,5,10,15,20,25,29`.
Projection sites: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
Ranks: `8,16,32,64,128`.

Two fits are intentionally measured:

### A1. Input-PCA projection

Fit right singular vectors of calibration input activations and execute the restricted operator `W P_r x`. This measures whether the current simple activation-subspace idea works directly.

### A2. Activation-aware reduced-rank regression

For calibration `X` and true operator outputs `Y=XW^T`, fit a rank-r output basis from `Y` and ridge-fit the latent map `X -> Z`. Evaluate the resulting rank-r `A(Bx)` on held-out activations.

This second test is the definitive low-rank gate because it directly optimizes the operator output rather than assuming PCA input directions are the right basis.

### Precommitted projection gate

Use held-out operator-output NMSE, not calibration energy:

- **pass rank32:** median NMSE <= 0.03 and >=70% of measured sites below 0.05 at rank 32;
- **pass rank64 only:** same criterion at rank 64 when rank 32 fails;
- otherwise **fail low-rank projection** as a universal core mechanism.

Failure does not imply every operator is incompressible. It means LARC must segment by operator/layer and/or use structured/sparse/rotated alternatives instead of extrapolating a universal low-rank basis.

## B. Raw real-layer interchangeability

Before recovery, physically alias representative real decoder layers to a neighboring layer and measure held-out autoregressive NLL. Also alias three small contiguous groups to one donor module.

This is deliberately harsh. It is a mechanism diagnostic, not the final sharing method.

Raw single-layer median perplexity ratio:

- <=1.10: surprisingly interchangeable;
- <=1.50: plausible recovery candidate;
- >1.50: raw exact sharing is severe and recovery/correction capacity is mandatory.

## C. Partial representation-consistent real conversion

Test a real four-layer group: logical layers `14,15,16,17`, physical donor layer `15`.

Baseline representation:

- all matrices canonical row-Q4;
- 1-D parameters FP16.

Student representation:

- identical baseline outside the group;
- all four logical layers reference one physical donor block;
- donor matrices use Q4_GROUP64;
- donor is hard-projected back to Q4_GROUP64 after every optimizer step;
- only donor parameters are trainable;
- recovery uses teacher-logit KL distillation on calibration windows disjoint from held-out evaluation.

### Precommitted partial-conversion gate

- **pass / expand:** post-recovery PPL ratio <=1.10 versus the row-Q4 teacher and group weight reduction >=3.5x;
- **borderline:** PPL ratio <=1.25 and group reduction >=3.0x;
- otherwise the current sharing recipe fails on this real group.

## D. Execution environment

The local analysis sandbox cannot follow Hugging Face's Xet/CDN redirect for the 269 MB checkpoint and cannot install Transformers from its restricted package index. Therefore the repository includes a GitHub Actions workflow that downloads the official checkpoint on a hosted runner and emits all Run-6 JSON artifacts.

The workflow does not change claim status merely by existing. Results become evidence only after a successful run is inspected and the generated artifacts are committed with provenance.

## E. Decision tree

1. Activation-aware rank32 passes + partial conversion passes: expand sharing to multiple real layer groups, integrate packed group64 weights + packed Q2/E4M3 KV, measure RSS.
2. Rank32 fails but rank64 passes: recompute the real memory budget at rank64 before any broader conversion.
3. Projection passes but sharing fails: preserve projection/KV work; reduce sharing aggressiveness or introduce depth-specific correction capacity.
4. Activation-aware projection fails broadly: stop treating low-rank projection as LARC's universal weight core and investigate structured/sparse/rotated alternatives.
5. Regardless of result, the next promoted L3 claim must compare against an actual optimized Q4_K_M/IQ baseline and use a standard evaluation corpus, not the built-in diagnostic prose alone.

## Current evidence boundary

Until the workflow successfully executes, Run 6 is **implemented protocol / pending real-checkpoint execution**, not L3 evidence.
