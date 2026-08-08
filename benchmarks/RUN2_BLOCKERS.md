# Run 2 external-validation blockers

This note records infrastructure failures so they are not confused with codec/model-quality failures.

## L3 checkpoint retrieval

Attempted external checkpoints included SmolLM2-135M and TinyStories-family checkpoints. Repository/model metadata and file manifests were reachable, but model payload downloads redirected to Hugging Face Xet objects that the current compute sandbox could not retrieve. A direct GitHub raw-binary route was also attempted for a committed TinyStories-15M INT4 binary; the compute container cannot resolve public GitHub hosts directly.

No external checkpoint conversion was executed. Therefore L3 is **unpassed**, not experimentally failed.

## GitHub Actions runner

A real-model GitHub Actions matrix was created as an alternative execution environment. Jobs failed before any workflow step was allocated. The workflow was then reduced to pure `run:` steps with no reusable actions and still failed before step execution. The workflow is now manual-only to avoid a permanently failing PR check.

## L4 accelerator hardware

The current execution environment exposes no CUDA/Metal accelerator. `runtime/triton_q4.py` therefore has source-level coverage only. GPU peak VRAM and throughput remain unmeasured.

## Required evidence to close blockers

1. Run `tools/real_model_benchmark.py` against an independently hosted pretrained checkpoint whose bytes are accessible to the execution host.
2. Benchmark the resulting LARC model against the named GGUF baseline at identical context length and generation settings.
3. Run the packed Triton/CUDA or future Metal runtime on actual accelerator hardware and capture peak allocated/reserved memory plus throughput.

These are evidence-access blockers only; they do not weaken or upgrade the existing L0/L1/L2/L2C results.
