# Run 19 — Medium-budget recursive recovery at the 10× capacity frontier

## Why this run is justified

Run 16's P=2/rank8 short-budget student is not usable: after 10,240 sampled training tokens its held-out PPL is still **97.23×** the teacher. It therefore fails the frozen Run-16 gate.

But global uptraining changed the student dramatically:

- initial student PPL ratio: ~76 million× teacher;
- final: 97.23×;
- first→best distillation loss reduction: **85.97%**.

That is qualitatively different from Runs 13–14, where the relevant FP32/local structural ceilings were already nearly flat failures. Run 16 shows substantial optimization recovery, just nowhere near enough under a 10k-token budget.

Run 19 therefore asks a precise next question: **does another 12.8× increase in sampled uptraining tokens materially close the held-out gap at the two adapter capacities that still fit the measured 10× description envelope?**

This is still tiny compared with billion-token recursive conversion work. It is a medium-budget conversion pilot, not a final architecture verdict.

## Frozen structural family

Physical recurrence phases stay fixed at **P=2**.

Logical projection:

`W_layer = B_(layer mod 2) + U_layer V_layer`.

Initialization and model topology are inherited from Run 16:

- shared full-rank base = arithmetic mean of assigned teacher matrices;
- depth adapter = randomized-SVD residual initialization;
- original per-layer RMSNorm stays unique and trainable;
- original tied embedding/head stays unchanged and frozen.

Two ranks:

### Rank 8

- exact main-projection parameter reduction: **11.152×**;
- Run-18 10× description envelope fits with Run-17 vocabulary PQ subdim≥12.

### Rank 16

- exact main-projection parameter reduction: **8.875×**;
- Q4_GROUP64 recursive decoder bytes: **6,588,288 B**;
- combined with Run-17 PQ24 + FP16 residual params + the full 1,855,488-byte conservative overhead allowance: **10,086,912 B = 10.455×** vs measured Q4_K_M;
- with PQ32: **9,792,000 B = 10.769×**.

Rank32 is not tested because it no longer fits the same current 10× description budget under the conservative allowance. This makes rank16 the maximum tested adapter capacity that is still directly relevant to the first headline milestone.

## Medium uptraining budget

Per arm:

- 512 optimizer steps;
- 4 independent random sequences/update;
- sequence length 64;
- **131,072 sampled training tokens** total;
- 12.8× Run 16 P=2 token budget.

Data remain disjoint:

- uptraining: WikiText-2 raw train;
- held-out quality: WikiText-2 raw test.

Objective remains:

`0.75 × forward-KL(student, teacher; T=2) + 0.25 × next-token CE`.

Optimization:

- shared bases LR 1e-4;
- depth adapters LR 5e-4;
- depth-specific norms LR 1e-4;
- no weight decay;
- 32-step linear warmup;
- cosine decay to 10% of initial LR by step512;
- gradient norm clipping at 1.0.

The four-sequence update reduces the extreme single-window noise visible in the short Run-16 loss trace while preserving the same teacher/student objective.

## Held-out evaluation

Every arm is evaluated end-to-end on **4,096** WikiText-2 raw test next-token predictions at context256, identically to Run 16.

The artifact records:

- exact structural parameter counts;
- description-budget compatibility;
- initial student short-probe PPL;
- final held-out teacher/student PPL and NLL delta;
- first/best training loss and CE;
- decayed-LR training history;
- for rank8, improvement relative to Run-16's 97.23× PPL ratio.

## Precommitted medium gate

**Pass medium recovery**:

- main-projection parameter reduction ≥8×; and
- held-out PPL ratio ≤**1.5×** teacher.

**Promising — extend to long uptraining**:

- reduction ≥8×;
- held-out PPL ratio ≤**10×** teacher;
- best training loss improves ≥50% from the first update.

Otherwise: `fail_medium_budget_only`.

As with Run 16, failure scope is explicit: it rejects only this **131k-token conversion recipe at the tested rank**. It does not falsify billion-token recursive uptraining or a student pretrained with recurrence from initialization.

## Decision value

- If rank8 passes or becomes promising, preserve its larger byte headroom.
- If rank16 is materially better and reaches the promising/pass region, prefer rank16 despite the smaller byte margin because it is still compatible with 10× when paired with PQ24/32.
- If both remain >10× teacher PPL after 131k tokens, the next structural move should not be another small local tweak. Either move to substantially larger/global uptraining data or start a recursively parameterized student from training initialization.

## Evidence boundary

Run 19 keeps the dense original tied embedding/head and does not quantize the recursive student. Exact parameter reductions and Run-18-compatible description envelopes are arithmetic, not a serialized `.larc` file. No native recursive runtime, RSS, VRAM, TTFT, or throughput claim is made.
