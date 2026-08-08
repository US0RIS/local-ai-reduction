# L3/L4 completion checklist

## L3 external pretrained model

- [ ] Acquire independent pretrained checkpoint bytes.
- [ ] Record named GGUF Q4 baseline and exact context.
- [ ] Measure baseline perplexity/NLL and generation/task quality.
- [ ] Convert after pretraining to LARC without requiring a second full dense local copy.
- [ ] Measure encoded file bytes and resident weights.
- [ ] Run packed latent-KV at identical context.
- [ ] Measure total inference-tensor memory.
- [ ] Demonstrate >=10x total reduction within the accepted quality threshold.
- [ ] Repeat on at least one 135M+ model before claiming general local-LLM applicability.

## L4 hardware

- [ ] CUDA packed-Q4 correctness test.
- [ ] CUDA latent-KV attention kernel.
- [ ] Peak allocated/reserved VRAM baseline measurement.
- [ ] Peak allocated/reserved VRAM LARC measurement.
- [ ] Tokens/s and time-to-first-token comparison.
- [ ] Same-context >=10x measured VRAM result.
- [ ] Metal implementation and unified-memory measurement, or explicitly scope the first hardware claim to CUDA.

No checkbox may be marked complete from modeled/synthetic evidence alone.
