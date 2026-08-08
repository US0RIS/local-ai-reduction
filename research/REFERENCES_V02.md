# LARC v0.2 additional prior art

These sources constrain the novelty and design claims for Run 2.

- **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache** (Liu et al., 2024). Empirically motivates per-channel key quantization and per-token value quantization; reports near-baseline quality on several LLM families with 2-bit KV. https://arxiv.org/abs/2402.02750
- **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** (Bae et al., ICLR 2025). Converts pretrained Transformers to repeated shared blocks and restores depth specialization with low-rank adapters. https://proceedings.iclr.cc/paper_files/paper/2025/hash/54d6a55225cebbdc16fbb0e45c5bdf2b-Abstract-Conference.html
- **Mixture-of-Recursions** (Bae et al., 2025). Combines recursive shared layers, adaptive token depth, selective KV caching, and a KV-sharing variant at 135M–1.7B scales. https://arxiv.org/abs/2507.10524
- **Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression** (Wang et al., ICLR 2025). Represents matrices across layers with shared bases plus unique coefficients and reports advantages at high compression ratios. https://proceedings.iclr.cc/paper_files/paper/2025/hash/238c98450b1d9e8055f94d22f303bb57-Abstract-Conference.html

LARC v0.2 does not claim invention of recursion, low-rank decomposition, low-bit KV quantization, paging, or packed low-bit kernels individually. The project hypothesis is that these mechanisms can be composed into one memory-budgeted storage/execution standard and that doing so can move complete local-inference memory materially beyond ordinary fixed-tensor GGUF deployment.
