# Run 9 — Competitive llama.cpp deployment baseline

## Purpose

Runs 6–8 falsified the broad post-training mechanisms that produced the controlled Run-5 compression ratio. Before designing another custom low-bit representation, LARC needs a real deployment target measured with a mature runtime and competitive GGUF quantizers.

Run 9 therefore measures **SmolLM2-135M** under current `llama.cpp` using:

- `Q4_K_M` as the named competitive Q4-class baseline;
- `Q2_K` as a mature low-bit comparison point;
- F16 GGUF as the quantization-quality reference.

This run does **not** test LARC.

## Source and versioning

The workflow checks out current `ggml-org/llama.cpp` and records the exact Git commit and version output in the result artifact. The SmolLM2 Hugging Face revision and SHA256 hashes of all generated GGUF files are also recorded.

Quantized GGUFs are generated from the same F16 GGUF with `llama-quantize`, with no importance matrix, so Q4_K_M and Q2_K are directly comparable under one conversion path.

## Quality

`llama-perplexity` is run over the complete WikiText-2 test corpus obtained using llama.cpp's `scripts/get-wikitext-2.sh` helper.

Fixed settings:

- context: 512;
- CPU only;
- F16 K/V cache;
- same corpus bytes for F16, Q4_K_M and Q2_K.

The final artifact records PPL and uncertainty plus ratios against F16 and Q4_K_M.

## Resident memory

For Q4_K_M and Q2_K, GNU `/usr/bin/time -v` measures process MaxRSS for a one-token `llama-cli` inference at allocated contexts:

- 64;
- 2048;
- 8192.

Each point is repeated three times in both default mmap and `--no-mmap` modes. The report stores every repetition and the median/min/max.

MaxRSS is a real process measurement on the GitHub-hosted Ubuntu CPU runner. It is not VRAM and must not be presented as a consumer-device L4 result.

## Throughput

`llama-bench` runs three repetitions with JSON output.

Prompt processing:

- pp64;
- pp2048;
- pp8192.

Token generation:

- tg128 after depth64;
- tg128 after depth2048;
- tg128 after depth8064 (chosen so 128 generated tokens remain within the model's 8192-token native context).

The benchmark uses CPU only and mmap enabled.

## Why this blocks the next codec

The project's simple row-Q4 research reference is not a competitive baseline on SmolLM2. A new LARC representation is only interesting if it improves materially on mature Q4_K_M/Q2_K tradeoffs under the same model, corpus, runtime class and context accounting.

Run 9 establishes:

1. the real Q4_K_M byte and RSS target that a 10× claim must beat;
2. the quality cost already incurred by mature Q2_K;
3. the amount of memory attributable to fixed runtime/KV overhead rather than model file bytes;
4. the CPU throughput cost of moving from Q4_K_M to a lower-bit representation.

The next custom weight experiment should be chosen only after these numbers are known.
