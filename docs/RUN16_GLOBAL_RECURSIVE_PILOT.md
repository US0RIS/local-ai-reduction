# Run 16 — Global recursive decoder distillation pilot

## Why Run 16 exists

Runs 6, 7, 13, and 14 all failed to recover extreme structural sharing with **local/post-hoc** fitting:

- exact whole-block sharing failed;
- shared output bases failed;
- full-rank shared bases + rank≤32 local residual factors failed even before factor quantization;
- a shared nonlinear MLP + tiny FiLM depth scales failed even after local function-space distillation.

Those failures do not test the strongest conversion recipe known for recursive Transformers: impose recurrence and then **globally uptrain the entire language model**, with depth-specific low-rank relaxation and teacher distillation.

Recent recursive-Transformer work uses moderate layer tying, SVD-initialized depth-wise LoRA, and billions of uptraining tokens. Run 16 borrows those principles but deliberately tests a much more aggressive sharing factor because the measured Run-12 file bound requires it.

This run is therefore a **short-budget pilot**. It can validate rapid recovery or reject this exact small-budget recipe; it cannot falsify long recursive uptraining.

## Student representation

All 30 logical decoder layers remain in the computation graph, but their seven large projection families are represented by only `P` physical recurrence phases:

`W_l = B_(l mod P) + U_l V_l`

for q/k/v/o/gate/up/down.

Two arms:

- `P=2`, rank8 depth adapters;
- `P=3`, rank8 depth adapters.

### Why rank8

For SmolLM2's seven main projection matrices, exact parameter arithmetic gives approximately:

- P=2/r8: **11.15×** projection-parameter reduction;
- P=3/r8: **8.13×**.

Ranks comparable to the 64–512 adapters used in moderate recursive-conversion literature surrender too much of the extreme compression objective on a 135M model. Run 16 intentionally tests only a compression-relevant adapter rank.

## Initialization

For every projection family and recurrence phase:

1. collect teacher matrices assigned to that phase (`layer mod P`);
2. initialize the shared full-rank base to their arithmetic mean;
3. compute each logical layer's residual from that mean;
4. initialize its rank8 adapter using randomized SVD of the residual.

This is materially stronger than zero-LoRA initialization and keeps the starting student as close to the pretrained teacher as the rank8 structural budget allows.

Original per-layer RMSNorm weights remain unique and trainable. Their storage is negligible relative to projection matrices and they provide an inexpensive depth-specific conditioning channel.

The original tied token embedding/head remains unchanged and frozen. Run 16 therefore isolates decoder-sharing viability; Run 15 separately tests the vocabulary-matrix floor.

## Global distillation

Unlike Runs 13–14, the complete student decoder is trained jointly through the language-model objective after recurrence is imposed.

Data:

- train/distillation: WikiText-2 raw train;
- held-out quality: WikiText-2 raw test;
- files are disjoint.

Budget per arm:

- 160 optimization steps;
- sequence length 64;
- at most 10,240 sampled training tokens;
- one random contiguous train window per step.

Objective:

`0.75 × forward-KL(student || teacher target distribution, T=2) + 0.25 × next-token CE`

Learning rates:

- shared full-rank bases: 1e-4;
- depth LoRA: 5e-4;
- layer-specific norms: 1e-4.

Teacher embeddings/head and student embeddings/head are fixed.

## Held-out quality

The final student is evaluated end-to-end for 4,096 next-token predictions from WikiText-2 raw test at context256. The teacher is evaluated identically.

The artifact records:

- initial student PPL on a 1,024-prediction probe;
- full teacher reference PPL;
- final student PPL and NLL delta;
- training loss history and best/final loss reduction;
- exact unique physical parameter counts in the shared PyTorch graph;
- SVD residual-initialization NMSE.

## Precommitted pilot gate

**Pass short-budget pilot**:

- decoder main-projection parameter reduction ≥8×; and
- final held-out PPL ≤1.5× teacher.

**Promising — extend uptraining**:

- reduction ≥8×;
- final PPL ≤3× teacher;
- best training loss improves by ≥40% relative to the first step.

Otherwise: `fail_short_budget_pilot_only`.

A failure is explicitly limited to this ≤10,240-token recipe. Published recursive conversions use orders of magnitude more uptraining data, so a failed pilot must not be written up as proof that recurrent/recursive pretraining cannot work.

## Research interpretation

If P=2/r8 or P=3/r8 recovers surprisingly well, the next step is longer distillation plus low-bit training of the small set of physical bases/adapters.

If both fail but training loss drops steeply, extend the budget rather than changing representation.

If both fail with little optimization progress, the next structural test should move recurrence into **joint architecture training** rather than conversion—e.g. a recurrent student trained from initialization with depth embeddings/LayerScale and eventually ternary/low-bit physical weights.

## Evidence boundary

Run 16 is a real-pretrained **full-language-model quality pilot** with original tied embedding/head left intact. It provides exact structural parameter counts but no serialized LARC file, no low-bit weight representation, no native recursive runtime, and no measured RSS/VRAM. Its held-out evaluation is a fixed 4K prediction slice, not the final full-corpus promotion benchmark.
