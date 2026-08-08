# Run 5 — reconciled response to the Run 4 audit

Run 5 was developed while `main` independently advanced the packed-runtime track. The final reconciliation therefore keeps **two evidence tracks** rather than overwriting one with the other:

1. **Packed E4M3 track (upstream Run 4):** native direct-packed latent-Q2 attention L1 evidence, disjoint calibration/evaluation, controlled context-64 quality, modeled >10× through 8K, but one training seed.
2. **Grouped multi-seed track (Run 5):** five-seed post-training conversion evidence with function-space prefit, group-64 QAT weights, group-3 latent-Q2 metadata and representation-matched quality, but Python/reference KV execution rather than the native packed kernel.

The next controlled milestone is to combine the best properties and rerun multi-seed quality through the native packed path.

## Audit dispositions

### Full-stack pairing

The Run-4 `2.45371` row came from Q4 weights with ordinary/full KV. It did not contain latent-Q2 KV. Run 5 therefore measures the grouped weight+KV stack directly rather than adding a historical KV delta.

### Correlated-error hypothesis

A six-realization stochastic-Q4 counterfactual gave only `+0.0343 ± 0.0482` nats/char average benefit from decorrelating error across depth, with two realizations getting worse. This does not support depth correlation as the primary mechanism.

Across four additional training seeds, teacher rows have mean absmax/RMS ~2.13 and raw row-Q4 matrix NMSE ~0.73%; the recovered shared block has absmax/RMS ~3.10–3.24 and row-Q4 NMSE ~1.56–1.73%. The stronger diagnosis is that the shared block's rows are intrinsically harder to quantize with one row scale.

Artifact: `benchmarks/run5_weight_diagnostics.json`.

### Weight fix

The selected reference codec uses one signed-Q4 FP16 scale per 64 contiguous weights. Shared-model modeled payload becomes 79,828 B versus ~77.3 KB for row-Q4, a small metadata increase that materially improves difficult shared weights.

### Conversion fix

Parameter averaging alone is poor. Current Run-5 reference conversion:

1. train 16-independent-block teacher;
2. initialize shared block from layer parameter mean;
3. 80-step **teacher-layer function prefit** over the union of all layer input/output transformations;
4. project to group-64 Q4;
5. 200-step LM recovery while hard-projecting matrix weights back to group-64 Q4 after every optimizer step.

Depth adapters and teacher-logit distillation were tested but did not improve the five-seed full-stack mean enough to promote.

### Metadata grouping

Run 5 treats token grouping as rate-distortion, not free compression. The selected reference profile shares one FP16 min/scale pair across both token and latent dimensions for each 3-token K group and V group. It retains and charges a worst-case FP16 incomplete-group tail.

### Context-dependent scratch

The grouped reference model uses `workspace(T)=3584+80T` B for the tiny controlled geometry. This is still structural/reference accounting, not measured RSS.

### K/V metrics

Both Q4 K and V bases use ridge-stabilized inverse Gram:

`(B B^T + lambda I)^-1`, with `lambda = 1e-5 * mean(diag(B B^T))` per head.

### Teacher-320

A naive constant-LR 320-step teacher degraded, so it is not a convergence ceiling. Tuned/decayed multi-seed convergence curves remain open.

### Tiny geometry vs SmolLM2

The controlled model is rank16/head-dim32 = 50%; the SmolLM2 planner is rank16/head-dim64 = 25% and uses GQA. Therefore the tiny model's KV asymptote is not an upper bound on SmolLM2 arithmetic.

## Run-5 grouped reference memory

Baseline: project row-Q4 teacher weights + FP16 KV + same reference workspace.

| context | reduction |
|---:|---:|
| 64 | **11.297×** |
| 128 | **11.195×** |
| 512 | **10.986×** |
| 2K | **10.887×** |
| 8K | **10.857×** |

Artifact: `benchmarks/run5_memory_context.json`.

## Run-5 grouped full-stack quality

Five training seeds `3,7,11,19,23`, 100,032 evaluation characters/seed.

Against the same project row-Q4 teacher representation used by this memory model:

- mean delta: **+0.03551 nats/char**;
- sample std: **0.16078**;
- mean perplexity ratio: **1.04705×**;
- ratio std: **0.17120**;
- range: **0.8969×–1.2363×**.

Against FP32 teacher, mean perplexity ratio is **1.37724×**.

Artifact: `benchmarks/run5_fullstack_multiseed.json`.

This is stronger than the upstream track on seed coverage but weaker on execution evidence: the grouped path is a representation-faithful Python/reference simulation, not the native `larc_q2_attention.cpp` packed kernel.

## Upstream packed track retained

Current `main` before reconciliation already contains:

- one-seed disjoint-stream controlled Q4/E4M3 latent result: perplexity ×**1.09392** at context 64;
- modeled packed-path reduction **12.04×** at context64 and **10.50×** at 8K;
- native direct packed-Q2 attention agreement to ~`2.5e-9` max abs vs decoded reference at T=2048.

These results remain valid and are not superseded by the grouped reference path.

## Current conclusion

The audit's practical recommendations were productive:

- dither was deprioritized after the mechanism test;
- finer weight scale grouping was validated as the more plausible fix;
- metadata grouping can restore long-context >10× in the reference model;
- five seeds exposed large conversion variance;
- function prefit + QAT reduced that variance dramatically.

But the project still lacks the single result that matters most next: **multi-seed controlled quality for the improved weight-recovery method running through a native direct-packed KV/weight runtime, followed by an actual Q4_K_M-class baseline and real pretrained model.**
