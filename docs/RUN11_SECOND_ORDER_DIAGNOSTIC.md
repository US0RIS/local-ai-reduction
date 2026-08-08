# Run 11 — Second-order, rotation, and outlier W2 diagnostic

## Motivation

Run 8 established that naive Euclidean/diagonal-RMS vector dictionaries are unusable as a primary extreme-weight codec on SmolLM2-135M. Run 11 does **not** try another dictionary. It asks whether the dominant real projection matrices become materially more 2-bit-friendly when the compressor uses information that Run 8 ignored.

The tested information is:

1. **block activation covariance** rather than only diagonal RMS;
2. **orthogonal activation/weight rotations** to alter quantization geometry without altering the exact FP32 operator;
3. **explicit outlier escape channels** selected by activation-weighted weight salience.

This is an operator-level diagnostic before any end-to-end codec promotion.

## Sites

Real SmolLM2-135M projections at layers `0,5,10,15,20,25,29`:

- q/k/v/o;
- gate/up/down.

This matches the depth sampling used in the real-model low-rank falsification work and includes the MLP matrices that dominate weight bytes.

## Calibration and held-out evaluation

- calibration: WikiText-2 raw **train** corpus;
- evaluation: WikiText-2 raw **test** corpus;
- four separated 512-token windows from each;
- 256 sampled calibration activation rows/site;
- 64 sampled held-out rows/site.

The workflow refuses to treat one file as both calibration and evaluation.

## Base W2 representation

The internal diagnostic W2 representation uses:

- four asymmetric levels;
- group size 64 along the input dimension;
- 2 dense code bits per weight;
- FP16 minimum + FP16 scale per output-row/group.

This is **2.5 bpw nominal including min/scale metadata** for full 64-element groups. It is a diagnostic representation, not claimed to match AutoRound or llama.cpp Q2_K semantics.

## Variants

### Plain Q2

Direct group64 min/max quantization.

### Block-GPTQ-style Q2

For each 64-input block:

- measure `H = X^T X / N` on calibration activations;
- apply 1% diagonal damping;
- invert the block Hessian;
- use the upper Cholesky factor of `H^-1` for sequential within-block quantization-error feedback.

The packed representation remains the same Q2 codes plus min/scale metadata. This implementation is labeled **GPTQ-style** rather than claiming parity with a particular external GPTQ package.

### Randomized Hadamard rotation

Each 64-input block gets a deterministic random sign diagonal followed by an implicit normalized Hadamard transform. The corresponding weight block is transformed identically so FP32 operator output is invariant before quantization.

Runtime metadata is only one sign bit/input channel because the Hadamard matrix itself is implicit.

### Learned orthogonal rotation ceiling

Each calibration covariance block is eigendecomposed and its eigenvectors are used as an arbitrary learned orthogonal transform. The workflow charges a full FP16 64×64 matrix per block.

This is intentionally an expensive **upper-bound diagnostic**. If it helps dramatically while Hadamard does not, the next research problem becomes finding a cheap structured approximation. It is not itself the preferred runtime representation.

### Outlier escape

Starting from the block-GPTQ Q2 representation, columns are ranked by:

`RMS(activation_column) × RMS(weight_column)`.

The top 1%, 2%, or 5% are restored exactly for the operator test. Byte accounting conservatively retains the full dense Q2 payload and adds:

- FP16 residual values for every escaped weight;
- a uint16 input-column index per escaped column.

The same escape test is also run after Hadamard + block-GPTQ.

## Metric

For every site/variant, held-out operator-output NMSE is:

`||X_hat Q^T - X W^T||² / ||X W^T||²`.

The artifact also records:

- encoded bytes;
- effective bpw;
- reduction versus FP16 for that matrix;
- weight and transformed-weight absmax/RMS diagnostics.

## Evidence boundary

Run 11 is **not** an end-to-end model result. It provides no model perplexity, task accuracy, native packed runtime, process RSS, or VRAM measurement. A variant can only advance to end-to-end testing if its held-out operator error is materially superior to plain W2 at comparable bytes, and the eventual choice must be interpreted alongside the independent Run-9 Q4_K_M/Q2_K and Run-10 AutoRound W2 references.
