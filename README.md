# local-ai-reduction / LARC

Research project for **LARC (Local Adaptive Representation & Compute)**: a new runtime-oriented local-AI model format intended to go materially beyond conventional dense GGUF quantization by storing model operators as shared activation-subspace cores plus progressive residual refinements.

## Target

Initial stretch target: **10–30× smaller stored/resident representation than a Q4-class GGUF baseline** while retaining useful model capability. This is a research target, not a current quality claim.

The central design principle is that the format must optimize the **stored representation, resident representation, and compute representation together**. A `.larc` runtime should not need to reconstruct full dense tensors before inference.

## Current v0.1 prototype

Implemented:

- minimal `.larc` container skeleton,
- row-wise Q4 and Q8 factor codecs,
- activation-subspace projection bundles shared by multiple linear operators,
- HRVQ64 progressive residual vector codec,
- synthetic compression/functional-error benchmarks,
- round-trip tests,
- technical specification and persistent action sheet.

Run 1 synthetic projection experiments reached **13.5×–23.1× compression vs the prototype Q4 baseline** in the intended low-rank regime. These are controlled synthetic operator tests, not proof that an actual LLM retains intelligence at those ratios.

See:

- [`ACTION_SHEET.md`](ACTION_SHEET.md) — detailed project log, benchmark interpretation, blockers, and next experiments.
- [`docs/SPEC.md`](docs/SPEC.md) — LARC v0.1 research specification.
- [`research/REFERENCES.md`](research/REFERENCES.md) — relevant prior art.

## Run locally

```bash
python -m pip install -e . pytest
pytest -q
PYTHONPATH=. python tools/benchmark_synthetic.py --n 768 --out benchmark_run1.json
PYTHONPATH=. python tools/benchmark_projection.py --n 384 --operators 5 --samples 768 --out benchmark_projection_run1.json
```

## Why not just make GGUF smaller?

llama.cpp already supports highly optimized low-bit weight formats. LARC is exploring a different abstraction: logical tensors may share learned bases and expose independently loadable quality refinements. That allows a runtime to trade fidelity for memory at a finer granularity than selecting one fixed quantization for every weight.
