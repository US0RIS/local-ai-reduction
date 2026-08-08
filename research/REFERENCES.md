# Research references guiding LARC v0.1

This list records techniques overlapping parts of LARC so the project does not mistake recombination for novelty. The research question is whether they can be integrated into a runtime/container contract that pushes model-level storage and resident-memory efficiency materially beyond ordinary GGUF quantization.

- **AQLM — Extreme Compression of Large Language Models via Additive Quantization** (Egiazarian et al., 2024): multi-codebook additive vector quantization around the 2–3 bit regime. https://arxiv.org/abs/2401.06118
- **QuIP#** (Tseng et al., ICML 2024): randomized Hadamard incoherence processing and E8 lattice vector codebooks. https://proceedings.mlr.press/v235/tseng24a.html
- **SqueezeLLM** (Kim et al., ICML 2024): sensitivity-aware quantization plus sparse treatment of outliers/sensitive weights. https://proceedings.mlr.press/v235/kim24f.html
- **LCQ** (Cai et al., 2024): low-rank codebooks for quantization. https://arxiv.org/abs/2405.20973
- **SVD-LLM** (Wang et al., 2024): truncation-aware data whitening and parameter updates for low-rank compression. https://arxiv.org/abs/2403.07378
- **CALDERA** (Saha et al., NeurIPS 2024): activation-weighted low-rank + low-precision decomposition `W ≈ Q + LR`, especially below 2.5 bits/parameter. https://arxiv.org/abs/2405.18886
- **ASVD** (Yuan et al., 2024/2025): activation-aware singular-value decomposition and layer sensitivity. https://openreview.net/forum?id=HyPofygOCT
- **BitStack** (Wang et al., 2024): progressive approximately 1-bit residual blocks for variable-memory sizing. https://arxiv.org/abs/2410.23918
- **bitnet.cpp / BitNet b1.58**: evidence that representation and low-bit kernels should be co-designed. https://github.com/microsoft/BitNet
- **llama.cpp / GGUF quantization**: practical baseline for local inference. https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
