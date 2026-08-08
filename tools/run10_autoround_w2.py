#!/usr/bin/env python3
"""Quantize SmolLM2 with a reproducible AutoRound W2A16G64 recipe.

Modes:
- rtn: pure round-to-nearest floor (no calibration optimization)
- tuned: AutoRound default-quality recipe plus the upstream INT2 algorithm extension

The full 1000-iteration/512-sample AutoRoundBest recipe is intentionally a later
escalation: Run 10 first determines whether competent 2-bit optimization moves the
small real model into a usable quality regime on a public CPU runner.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--mode", choices=("rtn", "tuned"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    args = ap.parse_args()

    from auto_round import AutoRound
    from huggingface_hub import model_info

    common = dict(
        model=args.model,
        scheme="W2A16G64",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    if args.mode == "rtn":
        recipe = dict(
            iters=0,
            nsamples=0,
            disable_opt_rtn=True,
            enable_alg_ext=False,
        )
    else:
        recipe = dict(
            iters=200,
            nsamples=128,
            seqlen=2048,
            batch_size=8,
            enable_alg_ext=True,
            disable_opt_rtn=False,
        )

    t0 = time.perf_counter()
    ar = AutoRound(**common, **recipe)
    args.out.mkdir(parents=True, exist_ok=True)
    ar.quantize_and_save(output_dir=str(args.out), format="auto_round")
    elapsed = time.perf_counter() - t0

    total_bytes = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    meta = {
        "mode": args.mode,
        "model": args.model,
        "source_model_commit": model_info(args.model).sha,
        "scheme": "W2A16G64",
        "format": "auto_round",
        "device_map": "cpu",
        "recipe": recipe,
        "wall_seconds_quantize_and_save": elapsed,
        "serialized_directory_bytes": total_bytes,
        "auto_round_version": __import__("auto_round").__version__ if hasattr(__import__("auto_round"), "__version__") else None,
        "runner": {
            "github_run_id": os.getenv("GITHUB_RUN_ID"),
            "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.getenv("RUNNER_NAME"),
            "runner_arch": os.getenv("RUNNER_ARCH"),
        },
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
