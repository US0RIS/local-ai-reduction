# Run 18 — Description-budget envelope

## Question

Run 12 proves that a 10× result versus the measured SmolLM2 Q4_K_M file must fit the **entire serialized model** in at most 10,545,369.6 bytes. Conventional same-parameter ≥1-bit quantization cannot do this.

Runs 16 and 17 test two structurally different components:

- Run 16: reduce the decoder's 106.17M main projection parameters to 2 or 3 physical recurrence phases plus rank8 depth adapters;
- Run 17: replace the 28.31M-parameter tied vocabulary matrix with direct-packed product quantization.

Run 18 asks only:

> If those pending representations pass quality, are their exact byte contracts sufficient to fit the measured 10× Q4_K_M file budget?

It is deterministic arithmetic, not a quality result.

## Conservative non-tensor allowance

Measured Run-9A values:

- F16 GGUF: 270,885,504 B;
- exact unique parameters: 134,515,008;
- raw unique FP16 tensor bytes: `134,515,008 × 2 = 269,030,016 B`.

The F16 GGUF therefore contains a surplus of:

**1,855,488 B**

above raw unique FP16 tensors.

Run 18 conservatively reserves this **entire** surplus for the future LARC file as an allowance for tokenizer data, container metadata, tensor descriptors, alignment/padding, and other non-modeled serialized overhead. LARC need not use GGUF metadata or this exact overhead; the allowance is deliberately not optimistic.

## Remaining non-projection parameters

SmolLM2 geometry:

- tied vocabulary matrix: 28,311,552 parameters;
- seven main decoder projections across 30 layers: 106,168,320;
- all other unique parameters: **35,136**.

All remaining parameters are charged at FP16:

**70,272 B**.

No saving is assumed for them.

## Recursive decoder byte contract

All physical shared projection bases and both rank8 LoRA factors are charged using the project's exact `Q4_GROUP64` byte formula: packed nibbles plus one FP16 scale per contiguous ≤64-value row group.

### P=2 physical recurrence phases, rank8

- physical shared bases: **3,760,128 B**;
- depth LoRA factors: **1,569,600 B**;
- total main-projection payload: **5,329,728 B**.

### P=3, rank8

- physical shared bases: **5,640,192 B**;
- depth LoRA: **1,569,600 B**;
- total: **7,209,792 B**.

These are representation bytes only. Run 16 must still establish whether the recursive model can preserve language-model quality.

## Direct-packed vocabulary PQ bytes

Run-17 representation:

- 294,912 B of FP16 codebooks at every subspace size;
- 98,304 B of FP16 token norms;
- one uint8 code per token/subspace.

| PQ subdim | vocab bytes | reduction vs tied Q4_GROUP64 |
|---:|---:|---:|
| 8 | 3,932,160 | 3.825× |
| 12 | 2,752,512 | 5.464× |
| 16 | 2,162,688 | 6.955× |
| 24 | 1,572,864 | 9.563× |
| 32 | 1,277,952 | 11.769× |

Again, these are bytes only; Run 17 owns quality and semantic direct-packed validation.

## Combined conservative description envelope

Every total below includes:

1. recursive decoder Q4_GROUP64 bases + Q4_GROUP64 rank8 depth LoRA;
2. Run-17 vocabulary PQ bytes;
3. all remaining 35,136 parameters at FP16;
4. the full **1,855,488-byte** conservative non-tensor allowance.

Measured comparison baseline: Q4_K_M GGUF = **105,453,696 B**.

10× target = **10,545,369.6 B**.

| physical phases | vocab subdim | conservative total | reduction vs Q4_K_M | 10× headroom |
|---:|---:|---:|---:|---:|
| 2 | 8 | 11,187,648 B | 9.426× | −642,278 B |
| **2** | **12** | **10,008,000 B** | **10.537×** | **+537,370 B** |
| 2 | 16 | 9,418,176 B | 11.197× | +1,127,194 B |
| 2 | 24 | 8,828,352 B | 11.945× | +1,717,018 B |
| 2 | 32 | 8,533,440 B | 12.358× | +2,011,930 B |
| 3 | 8 | 13,067,712 B | 8.070× | −2,522,342 B |
| 3 | 12 | 11,888,064 B | 8.871× | −1,342,694 B |
| 3 | 16 | 11,298,240 B | 9.334× | −752,870 B |
| 3 | 24 | 10,708,416 B | 9.848× | −163,046 B |
| **3** | **32** | **10,413,504 B** | **10.127×** | **+131,866 B** |

## Decision

**Serialized-byte economics are sufficient in principle for the first 10× milestone if the active Run-16/17 component representations pass quality.**

In particular:

- P=2/rank8 needs vocabulary PQ subdim12 or more aggressive;
- P=3/rank8 crosses 10× only at subdim32 under this conservative allowance.

This means the first 10× file milestone does **not** inherently require ternary or sub-1-bit physical decoder weights once major parameter sharing and vocabulary composition are present. Low-bit-trained physical weights remain highly relevant for extra headroom, resident memory, and the 20–30× objective, but they are not a prerequisite for the initial 10× byte envelope.

## Evidence boundary

Run 18 is arithmetic only. It assumes nothing about Run-16 recursive-model quality or Run-17 PQ vocabulary quality. `Q4_GROUP64` is not llama.cpp `Q4_K_M`; the comparison is an exact modeled candidate tensor contract plus a conservative overhead allowance versus the measured Q4_K_M serialized file. No `.larc` file exists yet, no recursive/PQ native runtime is integrated, and no RSS, VRAM, TTFT, or throughput saving is measured.

Generator: `tools/run18_description_budget.py`.
Canonical arithmetic artifact: `benchmarks/RUN18_DESCRIPTION_BUDGET.json`.
