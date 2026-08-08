# local-ai-reduction / LARC

**LARC (Local Adaptive Representation & Compute)** is an experimental local-AI model representation/runtime focused on reducing complete inference-memory cost, not merely model-file bytes.

## Target

This project's current target is **10×**, not maximum possible compression:

> Represent and execute a real pretrained model using no more than 10% of a named Q4-class deployment's relevant bytes at the same context, while retaining reasonable capability.

The target is **not yet proven on a real pretrained model or measured RAM/VRAM**.

## Current direction — Run 5: SoftShare-10X

Run 4 showed that exact/hard recursive sharing is too aggressive for a 10× goal: one Q4 block reused throughout depth suffers strongly correlated error and spends far less capacity on layer specialization than the byte budget requires.

The primary weight representation is now:

`W_(layer,type) = S_type + A_(layer,type) B_(layer,type)`

- `S_type`: one shared full-rank canonical-Q4 base per matrix type;
- `A B`: depth-specific low-rank Q4 residual;
- small layer-specific state retained;
- direct packed CPU execution evaluates `Sx + A(Bx)` without reconstructing dense per-layer weights;
- ranks and rescue pages are allowed to consume the budget up to the **10× boundary**.

Latent-Q2/E4M3 KV remains part of the design, but its rank is increased when the 10× budget permits.

## Controlled strategy evidence

Same 16-layer synthetic character-LM program used for prior controlled tests; canonical-Q4 teacher NLL **1.90547**:

| profile | complete-model reduction | final Q4 NLL | ppl ratio |
|---|---:|---:|---:|
| SoftShare rank 3/128 | **9.106×** | **1.98021** | **1.07760×** |
| uniform rank 2 | **10.142×** | **2.30033** | **1.48417×** |
| adaptive `qkv2/o1/fc1=3/fc2=2` | **10.046×** | **2.24237** | **1.40059×** |

The exact tiny-model 10× points have insufficient quality. The useful strategy signal is that rank3/128 = **2.34375% relative rank** preserves much more source information than hard recursion. On a 4096-hidden model that relative rank maps to rank96, where fixed overhead is far better amortized.

Artifact: `benchmarks/run5_softshare_control.json`.

## Real target: Mistral-7B-v0.1

Named Q4 deployment baseline:

- `mistral-7b-v0.1.Q4_K_M.gguf`
- exact size **4,368,438,912 bytes**
- SHA-256 `ce6253d2e91adea0c35924b38411b0434fa18fcb90c52980ce68187dbcbbe40c`

### Recommended starting core

Weight residual rank **96**, latent-KV rank **64**:

- modeled weight bytes: **369,495,040 B** = **11.8227×** smaller than the exact Q4_K_M file;
- modeled 4K KV: **44,105,728 B** vs **536,870,912 B** FP16 = **12.1724×**;
- weights + 4K KV: **11.8600×**;
- equal-common-scratch headroom before the structural ratio falls to 10×: **85,478,016 B**.

A higher-capacity rank128/KV72 candidate is still **10.6374×** structurally on weights+4K KV before common scratch.

These are exact byte calculations for the stated representations and exact named baseline. **No Mistral quality result exists yet.**

Artifact: `benchmarks/run5_mistral7b_budget.json`.

## Direct packed SoftShare — L1

`runtime/larc_q4.cpp` implements packed:

`y = Sx + A(Bx)`

with rank-sized scratch and no dense `W=S+AB` reconstruction.

Native correctness test:

- max abs error vs separately dequantized reference: **9.54e-7**;
- test residual rank: 23;
- scratch: 92 B.

Artifact: `benchmarks/run5_native_q4_softshare.json`.

## Streaming conversion

The Run-5 converter is designed so the complete source checkpoint never needs to exist locally at once:

- `LARCv2StreamWriter` writes compressed pages immediately;
- `larc/safetensors_range.py` reads individual tensors from local shards or exact HTTP byte ranges;
- remote servers must honor `Range` with HTTP 206/`Content-Range`;
- `tools/stream_softshare_convert.py` computes one shared matrix family at a time, then fits each layer residual against the **stored/dequantized Q4 shared base** and writes its factors immediately.

`tests/test_stream_softshare_convert.py` builds a two-shard synthetic SafeTensors checkpoint and verifies the resulting `.larc` page graph/CRCs. CI execution is pending runner allocation.

## Evidence / reproduction

- `ACTION_SHEET.md` — canonical technical status.
- `docs/RUN5_10X_PIVOT.md` — strategy rationale and gate definition.
- `benchmarks/INDEX.json` — artifact provenance registry.
- `benchmarks/RUN5_STATUS.json` — machine-readable current status.
- `tools/run5_softshare_control.py` — controlled SoftShare study.
- `tools/run5_budget_planner.py` — exact Mistral byte budget.
- `tools/stream_softshare_convert.py` — bounded-source-residency converter.
- `runtime/larc_q4.{h,cpp}` — packed Q4 and SoftShare primitives.
- `runtime/larc_q2_attention.{h,cpp}` — packed latent-Q2 attention.

## Still open

- real Mistral tensor access and L3 conversion;
- real-model perplexity/tasks/generation and adaptive validation-gain/byte rank allocation;
- long-context KV quality;
- final complete `.larc` <=10% of the named baseline;
- integrated measured CPU RSS / CUDA or Metal memory and throughput (L4);
- comparison against a 10× smaller dense/distilled model and other iso-byte alternatives.

**Do not state that LARC has demonstrated 10× lower measured RAM/VRAM or retained real-model intelligence yet.**
