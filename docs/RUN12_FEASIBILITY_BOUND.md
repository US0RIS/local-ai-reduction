# Run 12 — Exact Q4-relative feasibility bound

## Purpose

The project objective is **10–30× lower peak resident memory versus a competitive Q4-class deployment**, not merely a smaller FP16 checkpoint. Runs 6–8 show that the synthetic architecture does not transfer post hoc; Run 11 shows a real positive W2 mechanism, but only for part of the model.

Before spending more runs improving a same-parameter codec, Run 12 asks a mathematical question:

> Given the actual Run-9 Q4_K_M file, exact llama.cpp parameter count, and measured resident-memory pools, what information budget would a 10×, 20×, or 30× result actually require?

## Exact file bound

Once `run9_llamacpp_baseline.json` exists, the generator reads:

- exact Q4_K_M GGUF bytes;
- exact Q2_K and F16 GGUF bytes;
- `model_n_params` reported by `llama-bench`.

For target reduction `R`, the maximum total candidate file budget is exactly:

`B_candidate <= B_Q4 / R`.

The corresponding effective total-file bits/original-parameter is:

`bpw_required = 8 * B_Q4 / (R * N_params)`.

This is intentionally an **effective total-file** rate: container metadata, tensor scales, dictionaries, codebooks, residuals, tokenizer data, and any other serialized bytes all count against the budget.

If the 10× budget falls below one effective bit per original parameter, then an ordinary fixed-width same-parameter 1-bit codec cannot satisfy the file target even with zero metadata. At that point additional gain must come from parameter-count elimination/reuse, genuinely sub-bit entropy/structural coding, or a model trained for compressibility.

## Same-parameter zero-metadata bound

The script evaluates idealized 4, 3, 2.5, 2, 1.58, 1, 0.5, and 0.25 bpw payloads with **zero overhead**. These are intentionally impossible-best-case storage points for a conventional fixed-rate codec.

This bound is useful because it separates two research questions:

1. can better quantization improve quality at 2 bits? — Run 10/11;
2. can any same-parameter 2-bit representation reach 10× versus Q4? — Run 12 arithmetic.

The second question can be answered independently of model intelligence.

## Why Run 11 Q/K success cannot carry the whole objective

SmolLM2's seven main block projections have exact per-layer parameter fractions:

- Q: 9.375%;
- K: 3.125%;
- V: 3.125%;
- O: 9.375%;
- gate: 25%;
- up: 25%;
- down: 25%.

Therefore **Q+K are only 12.5%** of the main projection pool. Gate/up/down alone are **75%**.

Even an aggressive operator-adaptive scheme such as Q/K at 2.5 bpw and every other projection at 4 bpw averages 3.8125 bpw across this projection pool. That may be useful engineering, but it cannot explain a 10× Q4-relative whole-model result.

This is why Run 11 is a component result, not a path to the headline ratio by itself.

## Total resident-memory illustration

The generator also produces a deliberately labeled **illustrative** RSS model from Run 9. For each Q4 context/load-mode point it assumes serialized Q4 file bytes correspond one-for-one to removable resident weight bytes, with the remaining measured MaxRSS treated as a fixed floor.

This is not a measured decomposition. mmap residency, repacking, scratch allocation, kernel behavior, and allocator semantics can violate the assumption. The result is used only to show why fixed runtime/KV overhead can cap total-memory ratios, especially on a tiny 135M model.

The authoritative memory evidence remains Run 9's measured MaxRSS and llama.cpp-reported allocator pools.

## Decision use

Run 12 should determine which class of problem the project is actually solving:

- **same-parameter codec remains mathematically capable:** continue optimized low-bit/entropy work;
- **same-parameter fixed-rate codec cannot reach the target:** stop pretending another rounding trick can produce 10–30× and move the primary research effort to trained structural reuse, parameter elimination, recurrent/shared architecture learned from scratch/distillation, or genuinely sub-bit structured coding;
- **total RSS has a large fixed floor:** distinguish model-size reduction from total-memory reduction and test the goal on larger models where weights are a larger share of resident memory.

No quality claim is made by Run 12. It is a bound/decision artifact driven by the measured Run-9 baseline.
