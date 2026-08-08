# LARC Action Sheet

Canonical technical record for **LARC — Local Adaptive Representation & Compute**. Historical Run 1–3 detail is retained in `docs/RUN3_AUDIT_CORRECTIONS.md`; Run-4 hard-recursion evidence remains a reproducible reference/ablation. Artifact authority is `benchmarks/INDEX.json`; current machine status is `benchmarks/RUN5_STATUS.json`.

## Objective

The goal in this project is **10×**, not 20–50× and not maximum compression.

Acceptance target:

> A real pretrained model represented and executed with no more than 10% of a named Q4-class deployment's relevant bytes at the same context, while retaining reasonable capability.

Every promoted result must separate serialized bytes, resident weights, KV, scratch/workspace, measured RSS/device peak, context, quality, throughput, and evidence level.

Evidence levels: **L0** format; **L1** operator/runtime; **L2** controlled trained model; **L2C** controlled post-training conversion; **L3** independent pretrained model; **L4** measured target hardware.

---

# Why Run 5 changed direction

Run 4 made the methodology substantially stronger: representation-consistent Q4 quality, K/V inverse-Gram correction, disjoint calibration/evaluation, context-indexed accounting, packed latent-Q2 attention, and artifact provenance.

Its best hard-recursive controlled result remains useful: Q4 teacher NLL 1.88548 versus compressed NLL 1.97525, perplexity ×1.09392, with a composed 12.04× modeled context-64 tensor ratio.

But Run 4 also exposed the architectural weakness: forcing one Q4 block to perform every depth role creates strongly correlated error. Before Q4-aware recovery, the reused block produced roughly 2.19× teacher perplexity.

For a **10×** goal, that bottleneck is unnecessary. We can afford explicit layer-specific information.

## Primary architecture: SoftShare-10X

`W_(layer,type) = S_type + A_(layer,type) B_(layer,type)`

- one full-rank canonical-Q4 shared base `S_type` per matrix type;
- depth-specific low-rank canonical-Q4 residual factors;
- layer-specific small state retained;
- direct packed execution computes `Sx + A(Bx)` without dense reconstruction;
- residual ranks/refinement pages grow only until the complete deployment approaches the 10× boundary.

Hard recursion is now an ablation/fallback, not the preferred architecture.

---

# Run 5 — controlled strategy selection

Controlled model: d=128, H=4, FF=256, 16 independent teacher layers, context64, vocabulary37, 32,768 evaluation characters.

Canonical-Q4 teacher NLL: **1.90547344**.

## Representation-consistency correction

An early Run-5 experiment used 128-value grouped Q4 for A/B residual factors while the converter/runtime used canonical `Q4_ROW`. Its paired quality/byte results—including the earlier “10.142×” and “10.046×” tiny-model profiles—are **revoked**.

The authoritative rerun uses `Q4_ROW` for every residual-factor row, exactly matching the converter/runtime.

| profile | complete toy tensor reduction | final Q4 NLL | ppl ratio vs Q4 teacher |
|---|---:|---:|---:|
| uniform rank3 | **7.099×** | **1.85275** | **0.94864×** |
| uniform rank2 | **8.095×** | **1.98593** | **1.08378×** |
| uniform rank1 | **8.411×** | **1.91066** | **1.00520×** |

Artifact: `benchmarks/run5_softshare_control.json`; generator: `tools/run5_softshare_control.py`.

### Interpretation

This corrected experiment is stronger evidence for **quality preservation**, but weaker evidence about toy compression ratio.

At d=128, an A matrix with rank1–3 has only 1–3 values per row while `Q4_ROW` still spends two scale bytes per row. Consequently low-rank factor metadata dominates and the tiny model cannot serve as a complete-file 10× proxy.

The rank3 result slightly outperforms the finite-step Q4 teacher on this synthetic corpus after constrained recovery. This must not be interpreted as compression improving general intelligence: the teacher is small/undertrained and recovery provides extra distillation/training. It does show that explicit layer residuals survive the actual storage codec far better than hard tying.

The real hypothesis is scale-specific: `3/128 = 2.34375%` relative rank maps to rank96 at hidden size4096, where per-row scale overhead is small relative to factor payload.

---

# Run 5 — direct packed weight runtime (L1)

`runtime/larc_q4.{h,cpp}` implements direct packed:

`y = Sx + A(Bx)`.

Native correctness test (`tests/native_q4_softshare.cpp`):

- output shape 173×211, residual rank23;
- max absolute error vs separately dequantized reference: **9.53674316e-7**;
- rank scratch: **92 B**;
- no full `W=S+AB` reconstruction.

Artifact: `benchmarks/run5_native_q4_softshare.json`.

---

# Named real target: Mistral-7B-v0.1

Architecture used by the planner:

- 32 layers;
- hidden4096;
- intermediate14336;
- 32 attention heads;
- 8 KV heads;
- head dimension128;
- vocabulary32000;
- sliding window4096.

Named Q4 deployment baseline:

- `mistral-7b-v0.1.Q4_K_M.gguf`;
- exact size: **4,368,438,912 B**;
- SHA-256: `ce6253d2e91adea0c35924b38411b0434fa18fcb90c52980ce68187dbcbbe40c`;
- exact integer 10× file ceiling: **436,843,891 B**.

## Recommended starting core: weight rank96 / KV rank64

### Resident tensor model

- weight tensors: **369,495,040 B** = **11.82273×** vs exact Q4_K_M file;
- latent KV at 4096: **44,105,728 B** vs **536,870,912 B** FP16 = **12.17236×**;
- weights + 4K KV tensor ratio: **11.86001×**;
- equal-common-scratch headroom before that tensor ratio reaches 10×: **85,478,016 B**.

### Complete-file planning

The planner now charges the actual current `.larc` layout:

- 522 weight/state pages;
- Q4 page headers;
- manifest/page table;
- 4 KiB alignment;
- plus a conservative **4 MiB** tokenizer/config deployment reserve.

Rank96 estimate:

- serialized weight `.larc`: **371,302,608 B**;
- container/alignment/metadata overhead over tensor payload: about **1.80 MB**;
- with 4 MiB auxiliary reserve: **375,496,912 B**;
- conservative file ratio: **11.63375×**;
- remaining bytes before exact 10× file ceiling: **61,346,979 B**.

This margin is intentionally retained as **quality budget** for validation-selected rank/rescue pages.

The overhead-aware sweep demonstrates why complete-file accounting matters: uniform rank144 appears slightly above 10× on tensor bytes but falls to **9.964×** after container + auxiliary reserve, so it is rejected by the file target.

Artifact: `benchmarks/run5_mistral7b_budget.json`; generator: `tools/run5_budget_planner.py`.

**Mistral quality remains completely unvalidated.**

---

# Bounded-source-residency conversion

## Streaming `.larc` writer

`LARCv2StreamWriter` reserves manifest/page-table space, writes each compressed page immediately, retains fixed-size page records only, and patches the table/header at finalize. Output residency is O(current page + metadata), not O(output file).

## SafeTensors tensor-range input

`larc/safetensors_range.py` reads individual tensors from local shards or HTTP byte ranges. Remote sources must return HTTP `206` plus `Content-Range`; a server that ignores Range is rejected instead of silently transferring a complete multi-gigabyte shard.

Mistral's official source is two SafeTensors shards totaling **14,483,464,192 B** according to its index.

## Two-pass/per-family converter

`tools/stream_softshare_convert.py`:

1. streams one matrix family layer-by-layer to accumulate its shared base;
2. quantizes the shared base to canonical Q4;
3. discards the float precursor;
4. dequantizes the **stored** Q4 base;
5. rereads each layer tensor;
6. fits the low-rank residual against that stored base;
7. quantizes/writes A/B immediately;
8. releases source tensor and SVD workspace.

This preserves the representation-consistency rule learned in Runs 3–4.

## Local synthetic integration validation

GitHub has not allocated a workflow run for PR #8, so the branch's Python streaming path was reconstructed locally and run against the same intended test:

- two actual synthetic SafeTensors shards;
- two-layer Mistral-shaped graph;
- seven matrix families;
- rank2 residuals;
- **42 expected/actual `.larc` pages**;
- every page CRC verified;
- converter reports bounded source-file requirement = false.

Local report:

- bytes read from synthetic source/index: **18,832 B**;
- largest single source tensor: **768 B**;
- largest shared FP32 base: **1,536 B**.

This validates the mechanism, not real 14.5 GB Mistral conversion.

---

# Current claim boundary

The strongest defensible statement is:

> For a 10× objective, hard recursive tying is not the best current architecture. SoftShare retains explicit layer specialization and, with the exact Q4_ROW factor representation used by the converter/runtime, preserves controlled-model quality well (rank3 ppl×0.9486; rank2 ppl×1.0838), although tiny-model scale metadata prevents a meaningful complete 10× proxy. For the exact 4,368,438,912-byte Mistral Q4_K_M baseline, rank96/KV64 is structurally 11.860× on weight+4K-KV tensors and approximately 11.634× on the modeled serialized weight deployment after current container overhead plus a 4 MiB tokenizer/config reserve, leaving 61.35 MB of file budget before the exact 10× ceiling. Real Mistral quality is not yet measured.

Do **not** state that LARC has demonstrated retained Mistral intelligence, a real ≤436,843,891-byte Mistral `.larc`, or 10× measured RAM/VRAM.

## Next hard gates

1. **L3 source access:** run the range converter against the real Mistral SafeTensors shards.
2. **Real quality:** same-token perplexity/tasks/generation vs the named Q4_K_M baseline.
3. **Budget allocation:** spend the remaining ~61 MB complete-file margin only where real validation gain/byte is highest.
4. **Self-contained artifact:** embed/declare tokenizer/config resources and measure actual final `.larc` bytes.
5. **Long-context quality:** validate selected latent-KV rank at realistic context.
6. **L4 memory:** measure same-context CPU RSS and/or CUDA/Metal peak memory plus throughput.
7. **10× alternative control:** compare SoftShare against a genuinely ~10× smaller dense/distilled model and other feasible iso-byte representations.
