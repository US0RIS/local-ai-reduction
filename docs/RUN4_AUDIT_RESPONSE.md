# Run 4 audit response

This document records the disposition of the second external audit. It distinguishes accepted findings, findings already fixed in Run 3, and additional defects discovered while reproducing the audit.

## 1. Value-basis pseudo-inverse — accepted

The audit was correct. Q4 basis quantization breaks row orthonormality for V just as it does for K. Run 4 stores both FP16 inverse-Gram matrices and reconstructs the value aggregate with `v_lat G_V^-1 B_V_hat`. Basis and metric bytes are charged.

## 2. Equal-compute evaluation stream — accepted

The Run-3 equal-compute control did not use the same 100,032-character final stream as the L2C headline. Run 4 put teacher/continued-teacher/converted/scratch arms on one stream.

## 3. Equal-compute convergence interpretation — accepted; prior control revoked

Same-stream results exposed a more fundamental problem: continuing the independent teacher under the recovery schedule made NLL substantially worse. Therefore the old early-step control is not evidence that conversion wins at convergence. Stable tuned convergence curves, multiple seeds, and teacher-at-budget remain open.

## 4. Weight-quality representation consistency — accepted and decisive

Run 3 evaluated FP32 weights while charging Q4 bytes. Run 4 quantized both teacher and shared model to canonical Q4 for quality evaluation. Before Q4-aware recovery, repeated shared-block Q4 error was catastrophic (ppl ~2.19× the Q4 teacher). The Run-3 L2C headline is revoked.

Run 4 adds projected-Q4 recovery: after every optimizer update, all matrices are requantized to the exact Q4 storage grid. This reduces the Q4 shared-model penalty to ~0.0553 nats/char before KV compression.

## 5. Artifact/source divergence — accepted as systemic

The stale Run-2 native artifact is treated as a pipeline failure, not a one-off typo.

Added:

- `benchmarks/ARTIFACT_MANIFEST.json`,
- `tools/check_quick_benchmark_artifacts.py`,
- `.github/workflows/reproducibility.yml`,
- `tools/run4_l2c_repro.py`,
- `docs/RUN4_REPRO.md`.

Historical artifacts remain available for audit but are explicitly current/historical/superseded/revoked.

## 6. Q4 scale derivation — already corrected in Run 3; made normative

The audit correctly notes that `[-8,7]` alone does not define a quantizer. The current implementation already used:

`scale=max(max(row,0)/7, max(-row,0)/8, eps)`.

This is retained. Golden tests exercise both `-8` and `+7` codes. Therefore no switch to `absmax/7` or `absmax/8` was made.

## 7. Native rank floor — accepted and quantified

For `W=AB+sigma epsilon` with the generator's variance scaling, the rank-32 structural residual floor is `sigma²/(1/576+sigma²)`.

- sigma=.02: theoretical ~.18726; measured output NMSE vs exact W ~.26843.
- sigma=.002: theoretical ~.00230; measured ~.03330.

The second case is a cleaner factor-Q4 fidelity diagnostic.

## 8. KV metadata dominates — accepted; new codec added

Rank-16/head-dim-32 row Q2 spent 8 B/token on coefficients and 8 B/token on FP16 min/scale metadata across K+V. E4M3-FN metadata halves metadata to 4 B/token, yielding 12 B/token total and a raw KV ratio of 10.667× versus 128 B FP16 K+V.

At context64 on the clean controlled evaluation this metadata change does not materially worsen NLL.

## 9. Q4 basis row scales omitted — accepted

Current basis accounting includes packed nibbles, every FP16 row scale, and both FP16 inverse-Gram matrices.

## 10. Context sweep — accepted; restored as mandatory

The reference Python scratch path falls below 10× at long context despite FP8 metadata because it materializes an FP32 `T×r` latent history.

Run 4 implemented direct packed Q2 attention (`runtime/larc_q2_attention.cpp`) that never materializes that history. Its scratch is `T+4r` FP32 values/head. Under the direct-packed structural contract the modeled total remains ~10.60× at 2K and ~10.50× at 8K. **Only context64 has quality validation.**

## 11. Run-2 6.63× loose end — clarified

The 196,608-B term in `run2_recurrent_conformance.json` is not KV. It is `bounded_shared_scratch_bytes`, computed by the historical script as `4*64*128*6`. That experiment did not model KV.

## Additional defect found during Run 4

The first Run-4 latent-basis reruns fitted bases on the first contexts of the final evaluation stream. This is a compression-calibration leak. Those values were invalidated.

Current streams are disjoint:

- training seed 3,
- checkpoint-selection seed 444,
- latent-basis calibration seed 555,
- final-evaluation seed 333.

Current final 100,032-character result:

- Q4 teacher NLL 1.88548,
- Q4-recovered shared NLL 1.94078,
- shared + latent-Q2/E4M3 + both metrics NLL 1.97525,
- total +0.08977 nats/char,
- perplexity ×1.09392.

The direct-packed context64 tensor model is ~12.04×, but is composed L2C-quality + L1-runtime evidence rather than measured whole-process memory.

## Remaining hard gates

- stable converged equal-compute/multi-seed controls,
- real Transformer activation spectra,
- long-context quality,
- integrated packed full-model runtime + measured RSS,
- L3 independent pretrained model,
- L4 CUDA/Metal hardware,
- competitive iso-byte baselines,
- 20–30× real-model quality.
