# local-ai-reduction / LARC

**LARC (Local Adaptive Representation & Compute)** is an experimental local-AI representation/runtime focused on reducing complete deployment memory rather than model-file bytes alone.

## Target

This project's target is **10×**, not maximum compression:

> Represent and execute a real pretrained model using no more than 10% of a named Q4-class deployment's relevant bytes at the same context while retaining reasonable capability.

This target is **not yet proven on a real pretrained model or measured RAM/VRAM**.

## Run 5 direction: SoftShare-10X

Run 4 showed that exact/hard recursive sharing is unnecessarily severe for a 10× goal: one Q4 block reused throughout depth incurs strongly correlated error and discards layer specialization the byte budget can afford.

Primary weight form:

`W_(layer,type) = S_type + A_(layer,type) B_(layer,type)`

- shared full-rank canonical-Q4 base per matrix type;
- depth-specific low-rank canonical-Q4 residuals;
- layer-specific small state retained;
- direct packed execution evaluates `Sx + A(Bx)` without reconstructing dense per-layer weights;
- unused compression margin is **quality budget**, not an invitation to chase 20–50×.

Latent-Q2/E4M3 KV remains, with rank increased when the complete 10× budget permits.

## Controlled strategy evidence

Canonical-Q4 teacher NLL: **1.90547**. The authoritative Run-5 rerun uses exactly the `Q4_ROW` factor codec implemented by the converter/runtime:

| profile | complete tiny-model tensor reduction | final Q4 NLL | ppl ratio |
|---|---:|---:|---:|
| SoftShare rank3 | **7.099×** | **1.85275** | **0.94864×** |
| SoftShare rank2 | **8.095×** | **1.98593** | **1.08378×** |
| SoftShare rank1 | **8.411×** | **1.91066** | **1.00520×** |

An earlier Run-5 toy test used grouped-Q4 A/B factors while the real converter/runtime used `Q4_ROW`; its claimed ~9.1–10.1× toy ratios are **revoked**.

The corrected test says something narrower but useful: explicit layer residuals preserve controlled-model quality well under the actual codec. The d=128 toy is a poor complete-10× proxy because per-row scale metadata dominates 1–3-wide factors. Real-model quality remains the deciding experiment.

Artifact: `benchmarks/run5_softshare_control.json`.

## Real target: Mistral-7B-v0.1

Named baseline:

- `mistral-7b-v0.1.Q4_K_M.gguf`
- exact size **4,368,438,912 B**
- SHA-256 `ce6253d2e91adea0c35924b38411b0434fa18fcb90c52980ce68187dbcbbe40c`
- exact integer 10× file ceiling: **436,843,891 B**

### Recommended starting core: weight rank96 / KV rank64

Resident tensor model:

- weights **369,495,040 B** = **11.8227×** vs exact Q4_K_M;
- 4K KV **44,105,728 B** vs **536,870,912 B** FP16 = **12.1724×**;
- weights + 4K KV **11.8600×**;
- equal-common-scratch headroom before 10×: **85,478,016 B**.

Complete-file planning also charges current `.larc` page headers/table/alignment plus a conservative 4 MiB tokenizer/config reserve:

- serialized weight `.larc`: **371,302,608 B**;
- with auxiliary reserve: **375,496,912 B** = **11.6338×**;
- remaining bytes before exact 10× ceiling: **61,346,979 B**.

The overhead-aware planner rejects rank144 even though tensor payload alone appears slightly above 10×; after deployment overhead it is only ~9.964×.

These are structural/file-layout calculations. **No Mistral quality result exists yet.**

Artifact: `benchmarks/run5_mistral7b_budget.json`.

## Direct packed SoftShare — L1

`runtime/larc_q4.cpp` implements packed `Sx + A(Bx)` with rank-sized scratch and no dense `W=S+AB` reconstruction.

Native correctness: max abs error **9.54e-7** versus a separately dequantized reference.

## Bounded-source-residency conversion

- `LARCv2StreamWriter` writes compressed pages immediately;
- `larc/safetensors_range.py` reads individual SafeTensors tensors locally or through exact HTTP byte ranges and rejects servers that ignore Range;
- `tools/stream_softshare_convert.py` computes one shared matrix family at a time, quantizes the shared base, then fits every layer residual against that **stored/dequantized Q4 base** and writes factors immediately.

A local synthetic two-shard SafeTensors → `.larc` integration test passed: 42 expected/actual pages and all CRCs verified. This validates the mechanism, not real Mistral conversion.

## Evidence / reproduction

- `ACTION_SHEET.md` — canonical technical status
- `docs/RUN5_10X_PIVOT.md` — strategy rationale
- `benchmarks/INDEX.json` — artifact authority/provenance
- `benchmarks/RUN5_STATUS.json` — machine-readable status
- `tools/run5_softshare_control.py` — controlled strategy test
- `tools/run5_budget_planner.py` — Mistral 10× budget
- `tools/stream_softshare_convert.py` — range/shard converter
- `runtime/larc_q4.{h,cpp}` — packed Q4/SoftShare runtime
- `runtime/larc_q2_attention.{h,cpp}` — packed latent-Q2 attention

## Still open

- real Mistral tensor-range conversion (L3)
- real-model perplexity/tasks/generation and validation-gain/byte rank allocation
- actual self-contained `.larc` file measurement ≤436,843,891 B
- long-context KV quality
- measured CPU RSS / CUDA or Metal memory and throughput (L4)
- iso-byte comparison against a genuinely ~10× smaller dense/distilled model and other feasible alternatives

**Do not state that LARC has demonstrated 10× lower measured RAM/VRAM or retained real-model intelligence yet.**
