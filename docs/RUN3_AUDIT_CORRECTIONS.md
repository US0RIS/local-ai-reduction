# Run 3 audit corrections

Date: 2026-08-08

This note records an external technical audit of the Run 1/Run 2 evidence and the project's response. It is deliberately stricter than the prior status summary. Any claim contradicted by this note is downgraded until rerun.

## Confirmed material problems

1. **Native recurrent memory/quality pairing needs decomposition.** The L2 recurrent test's 14.61x weight-side reduction comes from representing one recurrently shared block once instead of physically duplicating it 16 times; the measured 11.53% NLL delta comes from KV compression on the already-shared model. The headline total-memory number must not imply that 11.53% NLL is the measured cost of both mechanisms. The weight-side transformation is exact only for a model whose logical definition already shares the block.
2. **Post-training +13.83% NLL comparison uses mismatched evaluation aggregation.** Teacher NLL is an eight-chunk average while packed-student NLL is one 64-token chunk. The L2C quality gate is revoked until rerun on identical tokens.
3. **End-to-end KV tests count bases as Q4 but execute FP32 PCA bases.** This means the joint memory/quality gates do not yet include basis quantization error. L2/L2C joint gates are provisional until rerun with the actual stored basis representation.
4. **Q4 scale dtype differs across implementations.** Python/Triton use FP16 scales; the native C++ benchmark stores FP32 scales. A format conformance pass must unify this and add golden-vector tests.
5. **Q4 encoder wastes code 0.** Encoders clip to [-7,7] while decoders support nibble-8 = [-8,7]. v0.3 should use the full signed nibble range with a range-aware row scale.
6. **Projection fitting composes against the pre-quantized basis.** `A=WU` is computed with float `U`, then inference uses quantized `U_hat`. Projection fitting should quantize the basis first and solve `min_A ||WX-A(U_hat^T X)||_F^2` on calibration activations.
7. **Quantized key bases require a metric correction.** If `B_hat` is not row-orthonormal, latent dot products should use `G^{-1}` where `G=B_hat B_hat^T`, or an equivalent whitening correction.
8. **HRVQ interpretation was too negative and codebook scope too vague.** At nominal 0.4375 bpw an i.i.d. Gaussian source has Shannon MSE bound `D/sigma^2 = 2^(-2R) ~= 0.545`; observed Gaussian NMSE 0.672 is only ~23% above that bound if codebooks are globally amortized. This is evidence that sub-0.5-bpw coding of unstructured Gaussian-like weights is fundamentally low-fidelity, not merely that this VQ implementation is poor. Conversely, if codebooks are per tensor, their 32,768 B/stage cost dominates and nominal bpw is not a valid compression claim. LARC must make codebook sharing/amortization explicit.
9. **Missing equal-compute control.** A one-block recurrent model trained from scratch for teacher-steps + recovery-steps is required to determine whether conversion is doing anything beyond selecting an oversized teacher for an easy task.
10. **Quality reporting is underpowered and uses a weak invariant.** Report delta nats/token and perplexity ratio, not only percent NLL; evaluate at least 100k held-out tokens/chars for controlled tests.
11. **SmolLM2 baseline mixing must be explicit.** Q4_K_M 105 MB is an external GGUF file baseline. Internal row-Q4 estimates are a different baseline and must never be interchanged.
12. **Embedding/head factorization is currently a high-risk unvalidated contributor** to SmolLM2-sized file reductions and requires rare-token/frequency-stratified evaluation.
13. **The Triton GEMV mapping is a reference contract, not a performance kernel.** One program per output row is likely suboptimal; split-K/reduction or tiled GEMV/GEMM should precede performance claims.

## Audit points checked and NOT confirmed as errors

### KIVI-latent 18.96x / 19.50x does use an FP16 reference

For SmolLM2 geometry in `memory_plan.py` (`kv_heads=3`, `head_dim=64`), FP16 K+V costs per layer/head/token:

`2 tensors * 64 dims * 2 B = 256 B`.

Rank-16 latent-Q2 asymptotic cost is:

- coefficients: `2 * 16 * 2/8 = 8 B/token`,
- value min+scale: `4 B/token`,
- key grouped metadata: `16*(2+2)/64 = 1 B/token`,
- two Q4 bases: 1024 B per layer/head, amortized as `1024/T`.

Thus cost is 13.5 B/token at T=2048 and 13.125 B/token at T=8192, giving exactly `256/13.5 = 18.963x` and `256/13.125 = 19.505x`. No FP32 baseline inflation is present in those two ratios.

### GQA is already represented in the SmolLM2 planner

`memory_plan.py` uses `kv_heads=3` and `head_dim=64`, not hidden width 576 for KV. GQA reduces absolute baseline and LARC KV bytes together; it does not by itself invalidate the reported KV compression ratio.

### The 6.63x recurrent-conformance result is reconstructible

It belongs to a different model (`d=64`) than the later `d=128` KV test. From `run2_recurrent_conformance.json`:

`(2,168,320 + 196,608) / (160,000 + 196,608) = 6.63173x`.

The prior prose should have made the configuration break explicit.

### The 7,680 B vs 8,704 B scratch difference is explained by latent rank

Both later models use `d=128`, `H=4`, `T=64`, `FF=256`, but the native recurrent test uses rank 12 and post-training conversion uses rank 16. Their shared scratch formula differs by `64*(16-12)*4 = 1024 B`, exactly matching 8,704 - 7,680.

### Native microbenchmark noise is not negligible

`W = A B + 0.02 epsilon` with `A_ij ~ N(0,1/R)` and `B_ij ~ N(0,1/K)` gives `Var(AB_ij) ~= 1/K = 1/576 ~= 0.001736`; noise variance is `0.02^2 = 0.0004`, about 23% of signal variance (about 19% of total variance), not ~1e-5. Therefore the 4.6% projected-vs-direct-Q4 output NMSE cannot be attributed purely to Q4 quantization.

## Revised evidence state

- L0 paged-container evidence remains valid.
- L1 native packed execution remains valid as a reference-kernel result, but byte accounting will be rerun after scale-dtype unification.
- L2 recursive graph aliasing remains valid.
- L2 joint memory+quality gate is **provisional** because the KV basis was not actually quantized and the headline pairing conflated exact architecture aliasing with KV degradation.
- L2C joint memory+quality gate is **revoked pending rerun** because of mismatched evaluation aggregation and unquantized-basis quality accounting.
- L3/L4 remain open.

## Run 3 priority order

1. Unify Q4 storage semantics across Python/C++/Triton and add golden-vector conformance.
2. Quantize KV bases for real, add key Gram-metric correction, and rerun L2.
3. Rerun L2C over identical >=100k validation tokens/chars, reporting delta nats and perplexity ratio.
4. Train an equal-compute one-block recurrent control from scratch.
5. Quantize-first + activation-weighted least-squares projection fitting.
6. Add attention-entropy sweep for latent-KV error.
7. Measure activation spectral energy on an independent real checkpoint as soon as checkpoint bytes are accessible.
8. Proceed to L3/L4 only after the corrected controlled gates pass.
