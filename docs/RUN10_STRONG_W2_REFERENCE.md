# Run 10 — Strong W2A16G64 post-training reference

## Purpose

Runs 6–8 show that LARC's previous low-rank, depth-sharing, and naive vector-codebook mechanisms do not transfer as broad post-training solutions to SmolLM2-135M. Run 10 asks a different question before another custom codec is designed:

> How well can a maintained, optimization-aware 2-bit PTQ system preserve this same pretrained model?

The reference implementation is Intel AutoRound **W2A16G64**, a real packed 2-bit weight-only format with CPU support.

## Compared representations

1. Original `HuggingFaceTB/SmolLM2-135M` checkpoint under the HF evaluator.
2. Pure RTN W2A16G64 — a no-calibration optimization floor.
3. Tuned AutoRound W2A16G64 — 200 iterations, 128 calibration samples, sequence length 2048, algorithm extension enabled, CPU tuning.

The tuned result is the relevant PTQ reference. RTN only tells us whether optimization materially changes the 2-bit regime.

## Why not immediately run AutoRoundBest?

Upstream recommends the 1000-iteration / 512-sample AutoRoundBest recipe for maximum 2-bit accuracy, but it is several times more expensive than the default tuned recipe. On a public CPU runner, Run 10 first determines whether the tuned W2 regime is even promising. If it materially improves over RTN and approaches a useful PPL ratio, the full best recipe becomes justified follow-up compute rather than blind expenditure.

## Evaluation

All three representations use the same Transformers/tokenizer path and the same canonical WikiText-2 raw test corpus.

- `add_special_tokens=False`;
- fixed context 512;
- non-overlapping token-stream windows;
- next-token cross entropy within each window;
- complete test corpus.

This creates a clean **within-runtime** PPL ratio for W2 vs the original checkpoint. Absolute PPL is not substituted directly for Run 9's llama.cpp PPL because evaluator details can differ.

## Serialization

The workflow records the complete serialized AutoRound model-directory byte count for RTN and tuned W2. This is useful storage evidence, but it is not equated with llama.cpp resident memory or a LARC runtime.

## Decision use

Run 9 and Run 10 together determine the next architecture direction:

- If tuned W2 retains substantially better quality than llama.cpp Q2_K and naive Run-8 codecs, future LARC weight work should start from second-order/optimized rounding, rotations, outlier channels, and mixed precision rather than new naive dictionaries.
- If even tuned W2A16G64 is unusable on SmolLM2-135M, >10× vs Q4 is unlikely to be obtainable by a simple post-training weight codec alone; training/distillation for compressible structure becomes a much stronger candidate.
- If tuned W2 is promising but still far from the target byte ratio, a hybrid path can combine strong W2-class weights with LARC's already validated packed latent-KV work.

No LARC claim is made by Run 10.
