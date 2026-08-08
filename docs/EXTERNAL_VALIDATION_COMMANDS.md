# External validation commands

Use these once checkpoint bytes and accelerator hardware are available.

## External pretrained CPU screening

```bash
python -m pip install torch transformers accelerate safetensors psutil sentencepiece hf_xet
PYTHONPATH=. python tools/real_model_benchmark.py \
  --profile 10x \
  --out benchmarks/external_10x.json
```

Repeat for `15x`, `20x`, and `30x`; only configurations that meet the recorded quality gate proceed to hardware validation.

## Native packed-Q4 CPU kernel

```bash
g++ -O3 -march=native -std=c++17 runtime/larc_q4.cpp tests/native_q4_smoke.cpp -o /tmp/larc-smoke
/tmp/larc-smoke

g++ -O3 -march=native -std=c++17 runtime/larc_q4.cpp tests/native_q4_bench.cpp -o /tmp/larc-bench
/tmp/larc-bench
```

## CUDA/Triton gate

The first GPU run must compare the same model/context with and without LARC and capture at minimum:

```python
torch.cuda.reset_peak_memory_stats()
# run warmup + measured generation
peak_allocated = torch.cuda.max_memory_allocated()
peak_reserved = torch.cuda.max_memory_reserved()
```

Also record tokens/s, time-to-first-token, context length, batch size, KV codec/rank, LARC page tier, and baseline GGUF/quantization identity.

The L4 gate is not satisfied by a kernel microbenchmark alone; it requires complete model generation and peak device memory.
