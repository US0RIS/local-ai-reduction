# Run 11 — Second-order, rotation, and outlier W2 diagnostic

## Motivation

Run 8 established that naive Euclidean/diagonal-RMS vector dictionaries are unusable as a primary extreme-weight codec on SmolLM2-135M. Run 11 does **not** try another dictionary. It asks whether the dominant real projection matrices become materially more 2-bit-friendly when the compressor uses information that Run 8 ignored.

The tested information is:

1. **block activation covariance** rather than only diagonal RMS;
2. **orthogonal activation/weight rotations** to alter quantization geometry without altering the exact FP32 operator;
3. **explicit outlier escape channels** selected by activation-weighted weight salience.

This is an operator-level diagnostic before any end-to-end codec promotion.

## Sites and data separation

Real SmolLM2-135M projections were tested at layers `0,5,10,15,20,25,29` for q/k/v/o/gate/up/down: **49 projection sites** total.

- calibration: WikiText-2 raw **train** corpus;
- evaluation: WikiText-2 raw **test** corpus;
- four separated 512-token windows from each;
- 256 sampled calibration activation rows/site;
- 64 sampled held-out rows/site.

The workflow refuses to treat one file as both calibration and evaluation.

## Base W2 representation

The internal diagnostic W2 representation uses four asymmetric levels, group size 64 along the input dimension, 2 dense code bits/weight, and FP16 minimum + FP16 scale per output-row/group. This is **2.5 bpw nominal including min/scale metadata** for full 64-element groups, or approximately **6.4× smaller than FP16 matrix storage**. It is a diagnostic representation, not claimed to match AutoRound or llama.cpp Q2_K semantics.

## Variants

### Plain Q2

Direct group64 min/max quantization.

### Block-GPTQ-style Q2

For each 64-input block, the harness measures `H = X^T X / N`, applies 1% diagonal damping, inverts the block Hessian, and uses the upper Cholesky factor of `H^-1` for sequential within-block quantization-error feedback. The packed representation remains the same Q2 codes plus min/scale metadata. This implementation is labeled **GPTQ-style** rather than claiming parity with a particular external GPTQ package.

### Randomized Hadamard rotation

Each 64-input block gets a deterministic random sign diagonal followed by an implicit normalized Hadamard transform. The corresponding weight block is transformed identically so FP32 operator output is invariant before quantization. Runtime metadata is only one sign bit/input channel because the Hadamard matrix itself is implicit.

### Learned orthogonal rotation ceiling

Each calibration covariance block is eigendecomposed and its eigenvectors are used as an arbitrary learned orthogonal transform. The workflow charges a full FP16 64×64 matrix per block. This is intentionally an expensive upper-bound diagnostic, not a proposed efficient runtime transform.

### Outlier escape

Starting from block-GPTQ Q2, input columns are ranked by `RMS(activation_column) × RMS(weight_column)`. The top 1%, 2%, or 5% are restored exactly. Byte accounting conservatively retains the full dense Q2 payload and adds FP16 residual values plus one uint16 input-column index per escaped column. The same experiment is run after Hadamard + block-GPTQ.

## Metric

For every site/variant, held-out operator-output NMSE is

`||X_hat Q^T - X W^T||² / ||X W^T||²`.

The artifact also records encoded bytes, effective bpw, reduction versus FP16, and weight/transformed-weight absmax/RMS diagnostics.

## Measured result

The real-model workflow completed successfully on SmolLM2-135M checkpoint `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

| representation | median held-out output NMSE | mean NMSE | fraction < 0.05 | median matrix reduction vs FP16 |
|---|---:|---:|---:|---:|
| plain Q2 | **0.16072** | 0.25129 | 6.12% | **6.400×** |
| block-GPTQ Q2 | 0.16527 | 0.23174 | 6.12% | **6.400×** |
| Hadamard Q2 | 0.15555 | 0.20998 | 6.12% | **6.396×** |
| **Hadamard + block-GPTQ Q2** | **0.09555** | **0.10649** | **36.73%** | **6.396×** |
| learned dense orthogonal Q2 | 0.15907 | 0.24625 | 6.12% | 3.740× |

The combined **Hadamard + second-order error-feedback** representation reduces median held-out operator NMSE by **40.55% versus plain Q2** while leaving the matrix byte ratio essentially unchanged. Neither ingredient alone produces the same gain, so the useful mechanism is the interaction between structured rotation and covariance-aware rounding/error feedback.

### Operator-family result

Median held-out NMSE for Hadamard + block-GPTQ Q2:

| operator | median NMSE |
|---|---:|
| Q | **0.03909** |
| K | **0.02994** |
| V | 0.19110 |
| O | 0.14008 |
| gate | 0.08547 |
| up | 0.15635 |
| down | 0.15705 |

This reinforces the prior Runs 6–7 observation that **Q/K are materially easier to compress than the dominant MLP/V/O byte pools**. Q/K are now plausible aggressive-W2 components. V, O, up, and down remain too inaccurate for an end-to-end all-W2 promotion.

### Outlier result

Outlier escape improves the weak unrotated GPTQ representation but does **not** improve the best Hadamard + GPTQ error/byte point:

- Hadamard+GPTQ, no escape: median NMSE **0.09555**, ~**6.396×** vs FP16;
- +1% FP16 columns: **0.10671**, ~5.995×;
- +2%: **0.10686**, ~5.642×;
- +5%: **0.10414**, ~4.836×.

Therefore the tested simple salience-based FP16 escape channel is **not promoted**.

### Learned-rotation result

The arbitrary covariance-eigenvector rotation achieves median NMSE 0.15907 and only 3.740× matrix reduction after charging FP16 transform matrices. This does not justify a learned dense-transform runtime. The positive rotation result is specifically the cheap structured Hadamard path.

## Decision

**Promote as a mechanism:** Hadamard rotation + block second-order/GPTQ-style rounding/error feedback.

**Do not promote as an end-to-end codec:** only 36.7% of sites are below 0.05 NMSE and the major V/O/up/down operators remain too inaccurate.

**Do not promote:** dense learned rotations or the tested 1–5% FP16 outlier escape channel.

The next end-to-end representation should be **operator-adaptive**, not uniform: it should test aggressive rotated/second-order W2 where Q/K tolerate it and preserve more capacity for V/O/MLP. The exact higher-bit/mixed strategy must be selected against the independent Run-9 Q4_K_M/Q2_K deployment baseline and Run-10 optimized AutoRound W2 reference.

## Provenance and evidence boundary

Generator: `tools/run11_second_order_diagnostic.py`.

Workflow run: `31278590079`; artifact ID `9027744924`; artifact digest `sha256:ef4145385d3ed5ce7761ba8ead540226bb0f3e9f85fe79cb15d4f61b83ee2192`.

Canonical committed summary: `benchmarks/RUN11_FINAL_STATUS.json`. The full per-site result is reproducibly generated and retained in the workflow artifact.

Run 11 is **not** an end-to-end model result. It provides no model perplexity, task accuracy, native packed runtime, process RSS, or VRAM measurement. No real-model LARC candidate is promoted by this run alone.
