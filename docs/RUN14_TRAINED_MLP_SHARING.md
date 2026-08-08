# Run 14 — Trained nonlinear MLP sharing

## Why this is a distinct hypothesis

Runs 6, 7, and 13 all start from independently trained layer matrices and then impose an algebraic sharing relation after the fact. Run 13 is especially informative: its full-rank shared base + rank≤32 additive residuals had excellent structural byte economics but failed even before residual-factor quantization.

Run 14 changes the object being fitted. Instead of approximating each **matrix**, it trains a shared **nonlinear MLP function** directly against real teacher MLP input/output pairs:

`x' = s_in[layer] ⊙ x`

`h = SiLU(G x') ⊙ (U x')`

`h' = s_mid[layer] ⊙ h`

`y = s_out[layer] ⊙ D h'`

`G`, `U`, and `D` are one physical full-rank MLP shared by several logical depths. The only per-layer state is three small FiLM-style elementwise scale vectors.

This tests whether shared structure can be **learned in function space**, rather than discovered as a low-rank linear difference between independently trained matrices.

## Why target the MLP first

For SmolLM2-135M's seven main decoder projections, gate/up/down account for **75%** of parameters. Q/K are only 12.5%. Run 11's positive Q/K low-bit result therefore cannot carry the original memory objective by itself.

A 10→1 MLP sharing mechanism attacks the dominant byte pool directly.

## Real-model protocol

Model: `HuggingFaceTB/SmolLM2-135M`.

Calibration pairs:

- WikiText-2 raw train;
- 128 real teacher MLP input/output activation rows per layer;
- two 512-token teacher windows.

Held-out pairs:

- disjoint WikiText-2 raw test;
- 64 rows/layer;
- two 512-token windows.

All 30 logical decoder MLPs are included.

Two sharing spans execute independently:

1. **10 logical layers / physical MLP** → 3 physical MLPs;
2. **5 logical layers / physical MLP** → 6 physical MLPs.

Each physical MLP is initialized from the arithmetic mean of the corresponding teacher gate/up/down matrices and then trained in function space for 120 balanced steps. Every update sees the same number of sampled activation rows from every logical layer in the group. The loss is the arithmetic mean of **per-layer normalized output MSE**, preventing high-output-energy depths from dominating the objective.

Shared full-rank matrices and depth-specific FiLM scales are both trainable. No teacher matrix is present in the student forward path.

## Representation-matched packed test

After local function distillation:

- physical gate/up/down matrices are hard-projected to `Q4_GROUP64`;
- all depth-specific FiLM scales are stored as FP16;
- no independent/dense logical MLP weights are counted.

The baseline is an independent `Q4_GROUP64` gate/up/down set for every one of the 30 teacher MLPs.

The artifact separately records:

1. untrained group-mean MLP + unit FiLM;
2. trained FP32 shared-function ceiling;
3. packed Q4 shared weights + FP16 FiLM.

Thus failure can be attributed either to the nonlinear sharing hypothesis itself or to the packed representation.

## Precommitted component gate

The gate uses **site-normalized per-logical-layer MLP-output NMSE**. It deliberately does not use Run 13's invalid raw cross-site energy sum.

**Pass** requires all of:

- MLP-pool byte reduction ≥ **8×**;
- packed median per-layer NMSE ≤ **0.05**;
- packed 90th-percentile NMSE ≤ **0.15**;
- worst physical-group median NMSE ≤ **0.10**.

**Borderline** requires:

- reduction ≥ **7×**;
- packed median ≤ **0.10**;
- packed p90 ≤ **0.25**;
- worst group median ≤ **0.15**.

The 5-layer sharing arm is expected to have less than 8× byte reduction and therefore cannot pass the headline component gate on byte economics alone; it is an important control showing whether reducing the sharing span materially improves function fidelity.

## Interpretation

A pass would justify replacing all teacher MLPs in an end-to-end SmolLM2 copy and evaluating WikiText-2 PPL before any runtime work.

A failure where the FP32 trained ceiling is already poor would be strong evidence that **local post-training function distillation is still insufficient**, pushing the structural route toward a model trained jointly with recurrent/shared depth from pretraining or broader distillation.

A good FP32 ceiling but poor packed result would instead direct effort back toward Run 11-style structured rotation/second-order quantization of the learned physical MLPs.

## Evidence boundary

This is an L3 real-pretrained **component** diagnostic, not a full model. It replaces only MLP functions in isolation. It makes no claim about attention, full-model perplexity, generation quality, runtime execution, process RSS, VRAM, or total-model compression.

Execution provenance: the complete harness, packed representation, data split, training hyperparameters, and site-normalized pass/borderline gate above were merged to `main` at `234eb325804d56a1c724103cfb7fcf7f6f4f9bf0` before this execution-only documentation change was opened.
