# LARC validation gates

The following gates must be satisfied before LARC is described as providing 10–30× lower VRAM for ordinary pretrained local LLMs.

## L3 — Independent pretrained-model conversion

Required:

- independently hosted pretrained model (first target: TinyStories 260K/15M or SmolLM2-135M; then 1B+),
- named GGUF baseline,
- identical context length and evaluation corpus,
- measured file and resident weight bytes,
- measured/allocated KV bytes,
- perplexity/NLL and external generation/task quality,
- ≥10× complete memory reduction at an accepted quality threshold.

The repository contains `tools/real_model_benchmark.py`, but current model payload retrieval is blocked by the execution host.

## L4 — Accelerator hardware

Required:

- CUDA and/or Metal implementation executes packed LARC representations without dense reconstruction,
- peak accelerator memory measured during real generation,
- baseline measured under identical context/generation settings,
- tokens/s and first-token latency recorded,
- ≥10× peak-memory reduction demonstrated for at least one useful external pretrained model before making the corresponding VRAM claim.

`runtime/triton_q4.py` is the current CUDA reference source; it has not been hardware validated.

## Current strongest evidence

The L2C controlled post-training conversion benchmark passes both current screening gates:

- 14.61× Q4-style weight reduction,
- 10.80× same-context complete inference-tensor reduction,
- +13.83% NLL versus a conventionally pretrained independent-layer teacher.

This is evidence that the representation/runtime strategy can cross the minimum target, not that the L3/L4 gates are already satisfied.
