# Next run: close L3/L4

The next substantive run should begin with the evidence blockers, not another synthetic codec experiment.

1. Acquire an independently pretrained checkpoint whose bytes are accessible to the execution host.
2. Run `tools/real_model_benchmark.py` and preserve baseline + converted outputs under `benchmarks/`.
3. If 10× quality fails, prioritize relaxed-recursive grouping, depth-specific low-rank adapters, and activation-weighted residual rescue rather than reducing rank blindly.
4. Run the resulting model through the packed CPU runtime and then on actual CUDA or Metal hardware.
5. Record peak accelerator memory and throughput at identical context settings.
6. Update `ACTION_SHEET.md` and `benchmarks/RUN2_FINAL_STATUS.json` (or create the next-run status artifact) only from measured evidence.
