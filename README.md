# local-ai-reduction / LARC

**LARC (Local Adaptive Representation & Compute)** is an experimental local-AI model storage and execution standard focused on reducing complete inference-memory cost rather than file size alone.

## Target

Research target: **10–30× lower peak resident inference memory than a named Q4-class baseline at the same context length while retaining useful capability.** This target is not yet proven on an external pretrained LLM or measured GPU VRAM.

## Audited Run 3 status

Run 3 corrected several Run-2 methodology and format issues identified by external review. The authoritative details are in `ACTION_SHEET.md` and `docs/RUN3_AUDIT_CORRECTIONS.md`.

The strongest corrected controlled L2C result trains a conventional 16-independent-block character LM first, converts it afterward to one recurrent/shared physical block, recovery-trains it, and evaluates an actually Q4-quantized latent-KV basis plus Q2 cache on the same fresh **100,032-character** held-out stream.

| metric | corrected result |
|---|---:|
| teacher NLL | 1.77359 |
| recovered shared student NLL | 1.88556 |
| compressed student NLL | 1.90953 |
| structural conversion delta | +0.11197 nats/char; perplexity ×1.11848 |
| KV compression delta | +0.02397 nats/char; perplexity ×1.02426 |
| total quality delta | **+0.13594 nats/char; perplexity ×1.14561** |
| modeled same-context inference-tensor reduction | **10.6628×** |

`10.6628×` is structural tensor accounting, **not measured process RAM/VRAM**. The task is a controlled synthetic character LM, not a general-LLM benchmark.

Artifact: `benchmarks/run3_posttrain_corrected_100k.json`.

An equal-compute control favored conversion/recovery over training the one-block recurrent model from scratch for the same 320 optimizer-step budget: recovered converted NLL 1.85563 vs scratch recurrent NLL 2.96558.

## Other Run 3 corrections

- Canonical Q4 now uses signed `[-8,7]`, all 16 nibble codes, low-nibble-first packing, and FP16 row scales across Python/native paths.
- The old Run-2 native operator artifact reporting `0.046151` NMSE is revoked as unreproducible from the checked-in source. The corrected synthetic rerun is much worse (~0.2684 NMSE vs exact FP32 W).
- Projection operators are fit against the **quantized basis actually executed** using calibration-weighted least squares.
- Latent key bases now include an FP16 inverse-Gram metric to correct non-orthogonality introduced by basis quantization.
- SmolLM2 rank-16 structural KV accounting, including the metric, is 18.62× smaller than FP16 KV at 2K and 19.41× at 8K. This is byte accounting only.

## Still open

- **L3:** independent pretrained 135M+ model conversion with standard perplexity/task evaluation.
- **L4:** measured peak CPU RAM / CUDA or Metal memory and throughput.
- Competitive iso-byte comparisons against actual GGUF/IQ, AQLM/QuIP#-class methods, and smaller dense models.
- Real activation-spectrum validation and rare-token validation of aggressive embedding/head factorization.
- Real-model 20–30× quality retention.

## Repository map

- `ACTION_SHEET.md` — canonical technical record.
- `docs/SPEC.md` — v0.3-candidate research specification.
- `docs/RUN3_AUDIT_CORRECTIONS.md` — detailed audit response.
- `benchmarks/RUN3_FINAL_STATUS.json` — machine-readable current status.
- `larc/paged_container.py` — paged mmap container.
- `larc/q4_runtime.py` — Python packed-Q4 reference.
- `larc/latent_kv.py` — latent-Q2 KV and basis-metric logic.
- `runtime/larc_q4.{h,cpp}` — native packed-Q4 CPU primitive.
- `runtime/triton_q4.py` — CUDA/Triton reference source; hardware validation open.

## Reproduction

```bash
python -m pip install -e . pytest torch
pytest -q
PYTHONPATH=. python tools/equal_compute_control.py

g++ -O3 -march=native -std=c++17 runtime/larc_q4.cpp tests/native_q4_smoke.cpp -o /tmp/larc-smoke
/tmp/larc-smoke

g++ -O3 -march=native -std=c++17 runtime/larc_q4.cpp tests/native_q4_bench.cpp -o /tmp/larc-bench
/tmp/larc-bench
```
