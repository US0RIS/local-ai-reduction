#!/usr/bin/env python3
"""Deterministic WikiText-2 perplexity for FP16 and packed AutoRound models.

This evaluator exists to compare AutoRound W2 variants against the original HF
checkpoint under one identical Transformers tokenization/execution path. Absolute
PPL is not asserted to be bit-for-bit identical to llama.cpp's evaluator; Run 9
owns the llama.cpp Q4_K_M/Q2_K baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tokenizer_name = args.tokenizer or args.model
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
    ).eval()

    text = args.corpus.read_text(errors="replace")
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if ids.numel() < 2:
        raise RuntimeError("Corpus tokenization is empty")

    total_nll = 0.0
    total_pred = 0
    chunks = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        # Non-overlapping fixed windows; every model compared here sees exactly
        # the same token IDs and chunk boundaries.
        for start in range(0, ids.numel() - 1, args.context):
            seq = ids[start:start + args.context + 1]
            if seq.numel() < 2:
                break
            x = seq[:-1].unsqueeze(0)
            y = seq[1:].unsqueeze(0)
            logits = model(input_ids=x, use_cache=False).logits.float()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                reduction="sum",
            )
            total_nll += float(loss)
            total_pred += int(y.numel())
            chunks += 1

    mean_nll = total_nll / total_pred
    out = {
        "model": args.model,
        "tokenizer": tokenizer_name,
        "corpus": str(args.corpus),
        "context": args.context,
        "tokenized_corpus_tokens": int(ids.numel()),
        "predicted_tokens": total_pred,
        "chunks": chunks,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "wall_seconds": time.perf_counter() - t0,
        "method": "non-overlapping HF token stream windows; add_special_tokens=False; next-token CE within each window",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
