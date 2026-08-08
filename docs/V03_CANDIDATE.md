# LARC v0.3 candidate — corrected low-bit semantics

This document supersedes the affected Q4 and latent-KV details in the older v0.2 spec while v0.3 remains experimental.

## Q4_ROW

For each row `x`:

`scale = max(max(x,0)/7, max(-x,0)/8, epsilon)`.

Store `scale` as IEEE binary16. Quantize with `q = clamp(round(x/scale), -8, 7)` and store `code=q+8`, low nibble first. Padding uses code 8 (zero). All cross-language implementations must match the golden vectors in `tests/test_q4_format.py`.

## Quantized latent bases

A stored basis is the dequantized Q4 matrix `B_hat`, not the float calibration basis. Because row orthogonality is not preserved by quantization, a conforming corrected latent-KV profile stores

`G_inv = (B_hat B_hat^T + ridge I)^-1`

in FP16 for every key and value basis/head or folds the equivalent transform into an adjacent stored operator.

### Key score

`q_lat = q B_hat_K^T`

`score(k) = q_lat^T G_K_inv k_lat / sqrt(d_head)`.

### Value reconstruction

After attention aggregation in latent space:

`v_full = v_lat G_V_inv B_hat_V`.

This is the pseudoinverse reconstruction and equals orthogonal projection onto the row space when the ridge tends to zero.

## Storage accounting

Every basis charge includes:

- packed Q4 coefficients;
- one FP16 scale per basis row;
- key inverse-Gram if not folded away;
- value inverse-Gram if not folded away.

Memory claims must state context length. A claim at context `T1` may not be promoted to another context `T2` without recomputing KV bytes and total bytes.

## Evidence rule

Quality measurements must execute the same weight and KV representation whose bytes are charged in the memory column. FP32-weight quality cannot be paired with Q4-weight memory as a complete compression-quality claim.
