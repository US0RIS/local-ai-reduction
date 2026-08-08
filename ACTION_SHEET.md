# LARC Action Sheet

Canonical technical record for **LARC — Local Adaptive Representation & Compute**. This file is updated after every substantive research run. Raw measurements are stored under `benchmarks/`; design rules are in `docs/SPEC.md`; the detailed external audit response is in `docs/RUN3_AUDIT_CORRECTIONS.md`.

## Objective

Build a local-AI storage/execution standard that can reduce **peak resident inference memory** by roughly **10–30× versus a named Q4-class deployment baseline at the same context length**, while retaining useful model capability.

Every result must distinguish:

- file bytes,
- unique resident weight bytes,
- KV-cache bytes,
- scratch/workspace,
- complete inference-tensor accounting,
- measured process/device peak memory,
- quality delta in nats/token and perplexity ratio,
- throughput/latency,
- baseline implementation,
- evidence level.

Evidence levels:

- **L0** — format/container integrity.
- **L1** — operator/kernel evidence.
- **L2** — controlled trained language-model conformance.
- **L2C** — conventionally parameterized model trained first, then structurally converted and recovered.
- **L3** — independently hosted pretrained LLM conversion.
- **L4** — measured target-hardware RAM/VRAM and throughput.

A lower evidence level must never be promoted as a higher one.

---

# Run 1 — representation feasibility

## HRVQ64

Implemented 64-weight residual vector coding with a scale shared across 256 weights and one 8-bit codebook index per 64-weight vector per stage.

Nominal payload rates excluding codebook transmission:

| stages | nominal bpw |
|---:|---:|
| 1 | 0.1875 |
| 2 | 0.3125 |
| 3 | 0.4375 |

Historical artifact: `benchmarks/benchmark_hrvq_run1.json`.

### Corrected Run-3 interpretation

At nominal rate `R=0.4375` bpw, the Gaussian squared-error rate-distortion lower bound is:

`D/sigma^2 = 2^(-2R) ~= 0.545`.

The measured Gaussian three-stage NMSE was ~0.672, only ~23% above that bound **if codebooks are globally amortized / treated as shared model state**. Therefore the main lesson is not merely that HRVQ implementation quality was poor; it is that sub-0.5-bpw coding of unstructured Gaussian-like weights is fundamentally too low-rate for high-fidelity reconstruction.

Codebook scope matters critically. A single 256×64 FP16 codebook is 32,768 B per stage. Three per-tensor stages cost 98,304 B before indices/scales. Therefore future HRVQ claims MUST state codebook sharing scope and include amortized codebook bytes. Nominal bpw is not a valid standalone compression claim for per-tensor codebooks.

**Current use:** residual/refinement candidate only, not a base representation.

## Activation-subspace projection

Run 1 tested five 384×384 operators under deliberately constructed activation covariance where the retained subspace contained 90/95/98% of activation energy.

Historical best cases:

- rank-10 Q4: 23.10× vs project row-Q4, ~0.026 output NMSE at 98% energy;
- rank-10 Q8: 13.47×, ~0.020 NMSE;
- rank-19 Q4: 13.47×, ~0.028 NMSE.

These are conditional synthetic results, not evidence that real LLM activations have this spectrum.

Run 3 corrected the fitting method; see below.

---

# Run 2 — complete-memory architecture prototype

Run 2 introduced:

- recursive/shared logical graph vs physical bundles,
- activation-subspace factors,
- latent 2-bit KV,
- packed Q4 CPU execution,
- Triton/CUDA reference source,
- v0.2 paged mmap container,
- controlled recurrent LM tests,
- post-training structural conversion,
- SmolLM2-135M benchmark harness.

## Results that survive audit

### L0 paged container

The v0.2 research container implements:

- 64-byte header,
- 64-byte page records,
- power-of-two aligned payload pages,
- mmap page views,
- per-page CRC32,
- dependency groups,
- shared/refinement/streamable/KV-basis flags,
- unique selected-page byte accounting.

Basic round-trip/alignment/checksum tests remain valid.

### Recursive graph aliasing

A recurrent LM with one physical block invoked at 16 logical depths was compared with 16 literal deep-copies of the same learned block. Validation NLL and logits were exactly identical in the test:

- max logit difference: 0,
- FP32 unique-weight reduction: 13.552×,
- reference Q4-style duplicate-vs-shared file reduction: 13.404×.

This proves lossless physical deduplication **for a model whose logical definition already reuses identical parameters**. It does not prove that 16 independently trained blocks can be replaced losslessly.

Artifact: `benchmarks/run2_recurrent_conformance.json`.

### Compressed-domain CPU execution

The native C++ kernel consumes packed nibbles inside the GEMV and computes projected `A(Bx)` using rank-sized scratch without reconstructing a dense `W=AB`.

That execution-property claim remains valid. Run 3 changed the canonical scale storage and reran performance/quality; the old numerical artifact is superseded.

### GPU source status

`runtime/triton_q4.py` is still source-only. No CUDA hardware has executed it. It is a reference contract rather than a performance result; one-program-per-output-row GEMV likely needs split-K/tiled redesign.

## Run-2 claims revoked/superseded by Run 3

1. `benchmarks/run2_native_q4_kernel.json` reported output NMSE `0.046151`; this does **not reproduce** from the checked-in Run-2 source and is revoked.
2. The native recurrent “11.53× total + 11.53% NLL” headline paired two different mechanisms. Weight reduction was lossless recurrent-block aliasing; measured degradation belonged to KV compression. Run 3 decomposes these explicitly.
3. End-to-end KV tests counted PCA bases using Q4 byte formulas but actually executed FP32 bases. Joint memory/quality gates were therefore provisional and have been rerun with actual quantized bases.
4. The post-training `+13.83% NLL` result compared an eight-chunk teacher average with one compressed-student chunk. It is revoked and replaced by a same-token 100k-character evaluation.
5. Run-2 C++ Q4 stored FP32 row scales while Python/Triton stored FP16 scales. Run 3 unifies the format.

---

# Run 3 — external audit corrections

## 1. Canonical Q4_ROW v0.3 candidate

Files changed:

- `larc/q4.py`
- `larc/q4_runtime.py`
- `runtime/larc_q4.h`
- `runtime/larc_q4.cpp`
- native tests
- `tests/test_q4_format.py`

Canonical rules now are:

- integer range: `[-8, 7]`, all 16 nibble codes reachable;
- stored code = `q + 8`;
- low nibble first;
- one IEEE-754 binary16 row scale;
- row scale = `max(max_positive/7, max_negative_magnitude/8)`;
- zero padding code = 8.

This fixes the old encoder/decoder mismatch where encoders emitted only `[-7,7]` while decoders accepted `[-8,7]`.

Golden vector shared by Python and C++ tests:

- row 1: `[-8,-4,0,3.5,7]`
- row 2: `[-1,0,1,2,3]`
- packed bytes: `40 c8 8f 86 da 8f` hex,
- FP16 scale bit patterns: `15360`, `14043`.

A local C++ compile confirmed the FP16 conversion/golden scale bits. Triton consumes the Python-format FP16 scale tensor but still requires hardware execution for L4 conformance.

## 2. Native projected-Q4 rerun

Configuration remains synthetic `1536×576`, rank 32, with:

`W = A B + 0.02 epsilon`.

Important audit correction: the noise is not negligible. Given `Var(A_ij)=1/R` and `Var(B_ij)=1/K`, `Var((AB)_ij) ~= 1/K = 1/576 ~= 0.001736`, while residual variance is `0.02^2 = 0.0004`, about 23% of low-rank signal variance.

Canonical-Q4 rerun:

- direct Q4 bytes: **445,440 B**,
- factor bytes: **36,928 B**,
- resident payload reduction: **12.062×**,
- direct scalar reference: ~452.2 µs,
- projected scalar reference: ~34.4 µs,
- speed ratio vs this scalar reference: **13.13×**,
- output NMSE vs direct-Q4 output: **0.28779**,
- output NMSE vs exact FP32 W output: **0.26843**,
- scratch: 128 B.

Recompiling the old Run-2 algorithm from the checked-in old source produced ~0.284 output NMSE, confirming that the archived `0.046151` value is stale/unreproducible.

Artifact: `benchmarks/run3_native_q4_kernel.json`.

**Interpretation:** packed-domain execution and structural compute reduction remain real; this particular synthetic operator has poor rank-32 approximation quality and is not evidence of LLM-layer fidelity.

## 3. Quantize-first activation-weighted projection fit

Old path:

1. fit float activation basis `U`;
2. compute `A = WU`;
3. quantize `U`;
4. inference uses `A(U_hat^T x)`.

Corrected path:

1. fit float activation basis;
2. quantize/dequantize to the actual stored `U_hat`;
3. form `Z = U_hat^T X`;
4. solve ridge-stabilized least squares:

`min_A ||W X - A Z||_F^2`;

5. quantize A.

At the previous rank-10 Q4 / 98%-energy / 23.10× point, corrected held-out output NMSE is **0.02483** vs old ~0.0260.

Artifact: `benchmarks/run3_projection_quantized_lstsq.json`.

This is a small but directionally correct gain. The dominant open question remains whether real Transformer activations permit ranks this small.

## 4. Quantized latent-KV bases and key metric correction

Run 3 now actually Q4-quantizes the learned per-head K/V bases before evaluation.

For dequantized key basis `B_hat`, latent coordinates alone induce metric `B_hat B_hat^T`. To approximate the orthogonal subspace dot product, LARC now stores an FP16 inverse-Gram correction:

`G_inv = (B_hat B_hat^T + ridge I)^-1`.

Query scoring uses:

`q_lat = q B_hat^T`

then

`score ~ q_lat^T G_inv k_lat / sqrt(d_head)`.

The correction matrix is included in storage accounting.

### KIVI-shaped structural accounting correction

The external audit suggested the previous 18.96×/19.50× values implied an FP32 baseline. That specific criticism is **not correct**.

For head_dim 64, FP16 K+V baseline is:

`2 * 64 * 2 = 256 B per layer/head/token`.

At rank 16 before Gram correction:

- K+V Q2 coefficients = 8 B/token,
- V FP16 min+scale = 4 B/token,
- grouped K metadata = 1 B/token,
- Q4 bases amortize as 1024/T B/token,

which gives 13.5 B at T=2048 and 13.125 B at T=8192, exactly 18.963× and 19.505×.

After charging the new FP16 key metric, rank-16 structural ratios become:

- **18.618× at 2K**,
- **19.412× at 8K**.

`memory_plan.py` already uses SmolLM2's `kv_heads=3`; GQA was not omitted.

## 5. Equal-compute control

A one-block recurrent model was trained from scratch for **320 optimizer steps**, equal to the teacher+recovery step count `120 + 200` in the post-training path.

Same model dimensions: d=128, H=4, FF=256, logical depth 16, context 64.

On 32 held-out contexts:

- independent-block teacher after 120 steps: NLL **1.71684**,
- converted shared model before recovery: **55.1402**,
- converted shared model after 200 recovery steps: **1.85563**,
- recurrent model trained from scratch for 320 steps: **2.96558**.

Teacher+recovery wall time in this run: ~31.18 s; scratch recurrent: ~34.31 s.

Artifact: `benchmarks/run3_equal_compute_control.json`.

**Interpretation:** on this synthetic task, conversion/recovery materially outperforms merely training the smaller recurrent architecture from scratch for the same optimizer-step budget. This weakens the simplest “teacher was pointless overcapacity” explanation, but does not address whether the teacher is converged enough or whether the result transfers to external LLMs.

## 6. Corrected native recurrent L2 screening

Configuration:

- d=128,
- H=4,
- logical depth=16,
- one physical recurrent block,
- context=64,
- latent rank=12.

The Q4 K/V bases are now actually executed; FP16 key inverse-Gram matrices are stored and counted.

Same-segment KV quality:

- baseline shared-model FP16-KV NLL: **2.01662**,
- Q2 latent-KV + Q4-basis NLL: **2.22656**,
- delta: **+0.20994 nats/char**,
- perplexity ratio: **1.2336×**.

Structural accounting:

- hypothetical 16 physical copies of identical block, Q4-style weights: 1,129,482 B,
- one shared physical block: 77,322 B,
- FP16 KV baseline: 524,288 B,
- packed Q2 KV: 57,344 B,
- Q4 bases + FP16 metric: 2,880 B,
- scratch: 7,680 B,
- baseline modeled tensors: 1,661,450 B,
- LARC modeled tensors: 145,226 B,
- modeled ratio: **11.440×**.

Artifact: `benchmarks/run3_recurrent_kv_corrected.json`.

**Claim boundary:** the 14.61× weight-side term is lossless physical aliasing of a recurrent model and carries zero measured quality cost by definition; the measured +0.20994 nats/char belongs to KV compression. The 64-char quality sample is still statistically weak and is screening evidence only.

## 7. Corrected post-training L2C result — strongest controlled evidence

The post-training experiment was rerun conceptually and numerically with the two main audit bugs fixed:

- teacher, recovered uncompressed student, and compressed student are evaluated on the **same fresh held-out stream**;
- learned K/V bases are actually Q4-quantized for execution;
- key inverse-Gram correction is stored as FP16 and counted;
- quality is reported as delta nats/char and perplexity ratio;
- long quality evaluation uses a dequantized-Q2 history fast path mathematically equivalent to literal pack/unpack; one-context logits matched the literal packed path within `6.68e-6` max absolute difference.

Training architecture:

- teacher: 16 independently parameterized blocks,
- teacher steps: 120,
- structural conversion: average teacher blocks into one recurrent physical block,
- recovery: 200 steps,
- latent rank: 16.

Independent calibration stream seed: 777.
Independent evaluation stream seed: 999.
Evaluation size: **100,032 characters**.

Quality:

| model | NLL |
|---|---:|
| independent teacher | **1.77359** |
| recovered shared student, uncompressed KV | **1.88556** |
| Q2 latent KV + Q4 bases + FP16 key metric | **1.90953** |

Decomposition:

- structural conversion/recovery cost: **+0.11197 nats/char**, perplexity ×**1.11848**;
- KV compression cost: **+0.02397 nats/char**, perplexity ×**1.02426**;
- total vs teacher: **+0.13594 nats/char**, perplexity ×**1.14561**;
- legacy percent-of-NLL representation: +7.66%, retained only for comparison; it is no longer the primary gate.

Memory model at context 64:

- teacher Q4-style weight bytes: 1,129,482,
- shared student Q4-style weight bytes: 77,322,
- weight structural ratio: 14.6075×,
- FP16 KV baseline: 524,288 B,
- packed latent-Q2 KV: 65,536 B,
- Q4 K/V bases + FP16 key metric: 4,352 B,
- KV ratio including basis/metric: **7.5018×**,
- scratch: 8,704 B,
- baseline modeled tensors: 1,662,474 B,
- LARC modeled tensors: 155,914 B,
- corrected modeled total ratio: **10.6628×**.

Artifact: `benchmarks/run3_posttrain_corrected_100k.json`.

### Current interpretation

This is now a legitimate controlled L2C result under the project's tensor-accounting model:

- a conventionally parameterized teacher is trained first;
- the model is structurally collapsed only afterward;
- recovery outperforms an equal-step scratch recurrent control;
- structural and KV quality costs are separately measured on identical 100k held-out characters;
- the KV basis representation used in quality is the representation charged in bytes;
- modeled same-context tensor accounting remains above 10×.

It is **still not measured process RAM/VRAM**, not a real external LLM, not a competitive GGUF/AQLM/QuIP# iso-byte comparison, and not evidence of broad reasoning/knowledge retention.

## 8. SmolLM2 structural planner after audit

The planner uses actual SmolLM2 GQA geometry (`kv_heads=3`, `head_dim=64`) and a named external Q4_K_M file baseline of 105 MB for file/weight-side comparisons.

With rank-16 latent KV including FP16 key metric:

| profile | context | LARC weight bytes | weight ratio vs 105 MB | KV ratio vs FP16 | modeled total ratio |
|---|---:|---:|---:|---:|---:|
| 10x | 2K | 8.06 MB | 13.02× | 18.62× | **13.94×** |
| 10x | 8K | 8.06 MB | 13.02× | 19.41× | **16.22×** |
| 15x | 2K | 5.31 MB | 19.78× | 18.62× | **18.62×** |
| 20x | 2K | 3.43 MB | 30.60× | 18.62× | **24.17×** |
| 30x | 2K | 2.43 MB | 43.23× | 18.62× | **28.73×** |

These remain **structural accounting only**. The major risk is not the extra metric bytes; it is whether ranks such as 32/576 preserve real activations and whether the aggressive rank-128 tied-vocabulary factorization preserves rare-token behavior.

## 9. Current evidence status after Run 3

| requirement | strongest current evidence | status |
|---|---|---|
| new random-access model format | v0.2 paged implementation | **L0 PASS** |
| canonical low-bit cross-language storage | Python+C++ golden Q4 semantics; Triton hardware unrun | **partial conformance** |
| direct packed CPU execution | native Q4 GEMV/projected GEMV | **L1 PASS for execution property** |
| operator fidelity at >10× | synthetic activation-subspace conditional tests; native noisy-rank32 test poor | **conditional only** |
| lossless shared-weight physical aliasing | recurrent identical-block conformance | **L2 representation PASS** |
| controlled post-training >10× tensor accounting | corrected L2C: **10.66×** | **PASS under modeled tensor accounting** |
| controlled quality at that point | +0.13594 nats/char; ppl ×1.1456 over 100,032 chars | **L2C screening PASS** |
| equal-compute smaller-model control | converted 1.856 NLL vs scratch recurrent 2.966 | **control favors conversion on synthetic task** |
| measured process RAM reduction | not measured | **OPEN** |
| independently pretrained external LLM | checkpoint bytes unavailable | **L3 OPEN** |
| real CUDA/Metal VRAM reduction | no accelerator available | **L4 OPEN** |
| 20–30× with retained real-model quality | not demonstrated | **OPEN** |

## 10. Highest-priority next work

1. **Finish Q4 conformance:** run Python/C++ golden test in CI and execute Triton against the same buffers on CUDA; add Metal equivalent.
2. **Update long-eval scripts** so the corrected 100k L2C path is the canonical reproducible program, not only a committed result artifact.
3. **Convergence control:** train the independent teacher substantially longer and repeat equal-compute / conversion sweeps across multiple seeds.
4. **Attention-entropy sweep:** quantify latent-KV output error versus attention entropy/peakiness; report V-error limit as attention becomes one-hot.
5. **Real activation spectra:** for any accessible Transformer checkpoint, capture per-site activation covariance and report cumulative energy at ranks 8/16/32/64/128 before attempting full conversion.
6. **Vocabulary risk:** frequency-stratified and rare-token reconstruction/logit error for tied embedding/head factorization; compare fallback strategies.
7. **Competitive baselines:** iso-byte comparisons with actual GGUF Q4_K_M/IQ variants, AQLM/QuIP#-class low-bit methods where runnable, and simply smaller dense models at the same resident bytes.
8. **Measured memory:** integrate packed Q4 weights and packed latent KV into one runtime and measure peak RSS before any VRAM claim.
9. **L3:** accessible 135M+ independent checkpoint, proper perplexity + LM-eval tasks, same-context comparison.
10. **L4:** CUDA/Metal peak memory, TTFT, tokens/s, and quality on the same L3 model.

## Current claim boundary

The strongest defensible statement after the audit is:

> LARC has an implemented paged format, direct packed CPU primitive, structural sharing semantics, quantize-first activation-subspace fitting, and a corrected controlled post-training experiment whose modeled same-context inference tensors are 10.66× smaller while held-out character-LM cross-entropy rises by 0.13594 nats/char (perplexity ×1.1456) over 100,032 independently generated characters. The result is L2C controlled evidence, not measured RAM/VRAM and not evidence for arbitrary pretrained GGUF models.

The project is **not entitled** to claim 10–30× less VRAM for real-world local LLMs until L3 and L4 pass.
